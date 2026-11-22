import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

// Load the renderer without WebGL and substitute the scene objects used by the tests.
function renderer(three = {}, fetch = globalThis.fetch) {
  const source = readFileSync(new URL('./renderers/skeleton.js', import.meta.url), 'utf8')
    .replace(/^import .*;\r?\n/gm, '')
    .replace('export const SkeletonRenderer', 'const SkeletonRenderer');
  return new Function('THREE', 'fetch', `${source}\nreturn { ...SkeletonRenderer };`)(three, fetch);
}

test('the Canvas opens live and cached joint streams with their own driver model', async () => {
  const extract = (file, name) => {
    const source = readFileSync(new URL(`./${file}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
    const start = source.indexOf(`function ${name}(`);
    assert.ok(start >= 0);
    return source.slice(start, source.indexOf('\n}\n', start) + 3);
  };
  const panelNode = () => ({ classList: { remove() {} }, style: {}, appendChild() {} });
  const mounted = [];
  const detail = new Function('mount', `
    const _cleanup = () => {}, openDetailPanelMobile = () => {}, _setStatus = () => {}, _connect = () => {};
    const node = ${panelNode.toString()};
    const _panel = node(), document = { getElementById: node, createElement: node };
    const ActivityRenderer = {}, TextRenderer = {}, RENDERERS = [{ canRender: () => true, mount }];
    const _CENTERED = '', _session = 0;
    let _textLike, _renderer, _frames, _status;
    ${extract('detail-panel.js', 'showTopicDetail')}
    return showTopicDetail;
  `)((_, mcpId) => mounted.push(mcpId));
  const cards = [{ el: { querySelectorAll: () => [] }, toolName: 'joints', id: 'left' },
    { el: { querySelectorAll: () => [] }, toolName: 'joints', id: 'right' }];
  const openDetail = new Function('_cards', 'showTopicDetail', 'fetch', `
    const inputArgs = () => ({}), _inputTopicsFor = () => [], _parseMcpCallResult = value => value;
    const _allMcps = [];
    async ${extract('canvas.js', '_openTopicDetailFor')}
    return _openTopicDetailFor;
  `)(cards, detail, async url => ({ json: async () => ({
    topic_out: url.includes('panther') ? [{ topic: '/panther/joints', format: 'sensor/skeleton' }] : [],
  }) }));

  await openDetail(cards[0].el, 'panther-driver', []);
  await openDetail(cards[1].el, 'quadruped-driver', [{ topic: '/quadruped/joints', format: 'sensor/skeleton' }]);
  assert.deepEqual(mounted, ['panther-driver', 'quadruped-driver']);

  const requested = [];
  const builds = [];
  const skeleton = renderer({}, async (url, init) => {
    requested.push([url, JSON.parse(init.body)]);
    const model = url.includes('panther') ? { urdf: '<robot/>', skeleton_visual_links: ['base_link'] }
      : { type: 'quadruped' };
    return { ok: true, json: async () => ({ code: 200, data: [{ type: 'text', text: JSON.stringify(model) }] }) };
  });
  skeleton._setModelStatus = () => {};
  skeleton._parseAndBuild = (...args) => builds.push(args);
  skeleton._buildQuadrupedSkeleton = model => builds.push(model);
  for (const mcpId of mounted) await skeleton._loadModel(mcpId, new AbortController().signal);
  assert.deepEqual(requested, mounted.map(id => [`/api/mcp/${id}/call?timeout_s=5`, { tool: 'model', arguments: {} }]));
  assert.deepEqual(builds, [['<robot/>', ['base_link']], { type: 'quadruped' }]);
});

test('a missing model is reported and a response after unmount is discarded', async () => {
  let response;
  const skeleton = renderer({}, () => new Promise(resolve => { response = resolve; }));
  const statuses = [];
  skeleton._setModelStatus = status => statuses.push(status);
  skeleton._parseAndBuild = () => assert.fail('no robot should be drawn');
  await skeleton._loadModel('dashboard', new AbortController().signal);
  assert.match(statuses.at(-1), /未找到/);
  const missing = skeleton._loadModel('robot', new AbortController().signal);
  response({ ok: true, json: async () => ({ code: 200, data: [] }) });
  await missing;
  assert.match(statuses.at(-1), /没有提供可用/);

  const request = skeleton._modelRequest = new AbortController();
  const late = skeleton._loadModel('robot', request.signal);
  skeleton.unmount();
  response({ ok: true, json: async () => ({ code: 200, data: { urdf: '<robot/>' } }) });
  await late;
  assert.equal(request.signal.aborted, true);
  assert.equal(statuses.at(-1), '正在加载机器人模型…');
});

test('successive lift readings replace the upper-body offset without moving its base', () => {
  const skeleton = renderer();
  const axis = { x: 0, y: 0, z: 1 };
  const position = {
    copy(v) { Object.assign(this, v); return this; },
    multiplyScalar(s) { this.x *= s; this.y *= s; this.z *= s; return this; },
  };
  const rotations = [];
  const base = { position: { x: 0, y: 0, z: 0 } };
  skeleton._joints = {
    body_joint: { userData: { axis, jointType: 'prismatic' }, position,
      quaternion: { setFromAxisAngle(...args) { rotations.push(args); } } },
    arm: { userData: { axis, jointType: 'revolute' },
      quaternion: { setFromAxisAngle(...args) { rotations.push(args); } } },
  };
  skeleton._links = { base_link: base };
  skeleton._loaded = true;
  const send = joints => skeleton.onData(new TextEncoder().encode(JSON.stringify({ joints })));
  send([{ name: 'body_joint', q: 0.62 }, { name: 'arm', q: 0.25 }]);
  assert.deepEqual([position.x, position.y, position.z], [0, 0, 0.62]);
  send([{ name: 'body_joint', q: 0.4 }]);
  send([{ name: 'body_joint', q: null }]);
  assert.deepEqual([position.x, position.y, position.z], [0, 0, 0.4]);
  assert.deepEqual(base.position, { x: 0, y: 0, z: 0 });
  assert.deepEqual(rotations, [[axis, 0.25]]);
});

test('base primitives render the box edges and a vertical rail at their URDF origins', () => {
  class Geometry {
    constructor(...args) { this.args = args; }
    dispose() { this.disposed = true; }
    rotateX(angle) { this.angle = angle; }
  }
  class Object3D {
    constructor(geometry) {
      this.geometry = geometry;
      this.position = { set: (...xyz) => { this.xyz = xyz; } };
      this.rotation = { set: () => {} };
    }
  }
  const skeleton = renderer({
    BoxGeometry: Geometry, EdgesGeometry: Geometry, CylinderGeometry: Geometry,
    Mesh: Object3D, LineSegments: Object3D, MeshPhongMaterial: class {}, LineBasicMaterial: class {},
  });
  const element = (attrs, elements = {}) => ({
    getAttribute: key => attrs[key] ?? null,
    querySelector: key => elements[key] ?? null,
  });
  const drawn = [];
  const link = { add: object => drawn.push(object) };
  skeleton._addPrimitiveVisual(link, element({}, {
    origin: element({ xyz: '0 0 0.1' }), 'geometry > box': element({ size: '0.65 0.51 0.2' }),
  }));
  skeleton._addPrimitiveVisual(link, element({}, {
    origin: element({ xyz: '0.163 0 0.803' }), 'geometry > cylinder': element({ radius: '0.008', length: '1.206' }),
  }));
  assert.equal(drawn.length, 2);
  assert.deepEqual(drawn[0].geometry.args[0].args, [0.65, 0.51, 0.2]);
  assert.deepEqual(drawn[0].xyz, [0, 0, 0.1]);
  assert.equal(drawn[1].geometry.angle, Math.PI / 2);
  assert.deepEqual(drawn[1].xyz, [0.163, 0, 0.803]);
});
