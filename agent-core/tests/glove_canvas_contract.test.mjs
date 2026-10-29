import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const canvas = readFileSync(new URL('../web/js/canvas.js', import.meta.url), 'utf8');
const skeleton = readFileSync(new URL('../web/js/renderers/skeleton.js', import.meta.url), 'utf8');

function functionSource(name) {
  const start = canvas.indexOf(`async function ${name}(`);
  assert.ok(start >= 0);
  return canvas.slice(start, canvas.indexOf('\n}\n', start) + 2);
}

test('live stream details use the producing driver to fetch its model', async () => {
  const el = {};
  const calls = [];
  const context = {
    _cards: [{ el, id: 'card-left', toolName: 'glove_teleop' }],
    inputArgs: () => ({}), _inputTopicsFor: () => [],
    fetch: async () => ({ json: async () => ({ topic_out: [{ topic: '/hands', format: 'sensor/skeleton' }] }) }),
    _parseMcpCallResult: value => value,
    showTopicDetail: (...args) => calls.push(args),
    console,
  };
  const open = vm.runInNewContext(`(${functionSource('_openTopicDetailFor')})`, context);
  await open(el, 'mcp-glove', []);
  assert.deepEqual(calls, [['/hands', 'sensor/skeleton', 'mcp-glove']]);
});

test('cached stream details retain the same model provider', async () => {
  const calls = [];
  const context = {
    _cards: [], _allMcps: [], console,
    showTopicDetail: (...args) => calls.push(args),
    _showToast: () => assert.fail('a cached topic exists'),
  };
  const el = { querySelectorAll: () => [] };
  const open = vm.runInNewContext(`(${functionSource('_openTopicDetailFor')})`, context);
  await open(el, 'mcp-glove', [{ topic: '/hands', format: 'sensor/skeleton' }]);
  assert.deepEqual(calls, [['/hands', 'sensor/skeleton', 'mcp-glove']]);
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
