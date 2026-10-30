// Exercise the real canvas info-response and drag handlers with a small DOM stub.
// This is not a substitute for a real-browser interaction test.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('./canvas.js', import.meta.url), 'utf8');
const canvas = source
  .replace(/^import .*? from ['"].*?['"];\s*$/gm, '')
  .replace(/^import\s*\{[\s\S]*?\}\s*from\s*['"].*?['"];\s*$/gm, '')
  .replace(/^export /gm, '');

function harness() {
  const handlers = {};
  const viewportHandlers = {};
  const windowHandlers = {};
  class Element {
    constructor() {
      this.dataset = {};
      this.style = {};
      this.isConnected = true;
      this.classList = { add() {}, remove() {}, toggle() {} };
      this._ports = [];
      this._listeners = {};
    }
    set innerHTML(html) {
      this._ports = [...html.matchAll(/<div class="canvas-port (in|out)[^\"]*"([^>]*)>/g)].map((match) => {
        const p = new Element();
        p.direction = match[1];
        for (const [, name, value] of match[2].matchAll(/data-([\w-]+)="([^"]*)"/g)) {
          p.dataset[name.replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = value;
        }
        p.closest = () => this;
        return p;
      });
    }
    querySelector(selector) {
      if (selector.includes('data-idx=')) return this._ports.find(p => p.dataset.idx === selector.match(/data-idx="([^"]*)"/)[1] && (!selector.includes('.in') || p.direction === 'in') && (!selector.includes('.out') || p.direction === 'out')) || null;
      if (selector === '.canvas-card-instance-cfg-btn' || selector === '.canvas-exec-btn') return null;
      return new Element();
    }
    querySelectorAll(selector) { return selector === '.canvas-port.out' ? this._ports.filter(p => p.direction === 'out') : selector === '.canvas-port.in' ? this._ports.filter(p => p.direction === 'in') : []; }
    addEventListener(name, fn) { this._listeners[name] = fn; }
    replaceWith(other) { this.isConnected = false; for (const p of this._ports) p.isConnected = false; this.replacement = other; }
    getBoundingClientRect() { return { left: 0, top: 0, width: 10, height: 10 }; }
    appendChild() {}
    remove() {}
    setAttribute() {}
    closest() { return this; }
  }
  const document = {
    createElement: () => new Element(),
    createElementNS: () => new Element(),
    addEventListener: (type, fn) => { handlers[type] = fn; },
    elementFromPoint: () => null,
    getElementById: () => null,
  };
  const viewport = new Element();
  viewport.addEventListener = (type, fn) => { viewportHandlers[type] = fn; };
  let dropTarget = null;
  document.elementFromPoint = () => dropTarget;
  const context = vm.createContext({
    document, window: { addEventListener(type, fn) { windowHandlers[type] = fn; } }, console,
    sessionId: () => 'test', inputKey: topics => JSON.stringify(topics), inputArgs: () => ({}), showToast() {},
    setTimeout: () => 1, setInterval: () => 1, clearTimeout() {}, requestAnimationFrame: () => 1,
    fetch: async () => ({ json: async () => ({ code: 200, data: { topic_out: context.live } }) }),
  });
  vm.runInContext(canvas, context, { filename: 'canvas.js' });
  context.viewport = viewport;
  context.svg = new Element();
  vm.runInContext('_viewport = viewport; _connSvg = svg; _setupPortDrag()', context);
  return { context, handlers, windowHandlers, viewportHandlers, Element, setDropTarget(target) { dropTarget = target; } };
}

function setup(h, saved, live) {
  h.context.saved = saved;
  h.context.live = live;
  vm.runInContext(`
    _allMcps = [{id:'g1', tools:[{name:'lidar_cloud', type:'sensor', topic_out:live}]}];
    const el = _buildCardEl({id:'source',mcpId:'g1',toolName:'lidar_cloud',x:0,y:0,topicOut:saved});
    _cards = [{id:'source',mcpId:'g1',toolName:'lidar_cloud',x:0,y:0,el,topicIn:[],topicOut:saved}];
  `, h.context);
}
const legacy = [{topic:'/legacy',format:'sensor/pointcloud'}];
const current = [...legacy, {topic:'/nav/cloud',format:'sensor/pointcloud',port:'navigation_lidar'}];

async function beginWire(h) {
  vm.runInContext('_projectStateKnown = true; _projectRunning = false; _isEditor = true', h.context);
  const sourcePort = vm.runInContext('_cards[0].el.querySelectorAll(".canvas-port.out")[1]', h.context);
  sourcePort.closest = selector => selector === '.canvas-card' ? vm.runInContext('_cards[0].el', h.context) : sourcePort;
  const targetCard = new h.Element();
  targetCard.dataset.cardId = 'destination';
  const targetPort = new h.Element();
  targetPort.dataset = {format:'sensor/pointcloud', idx:'0', port:'lidar'};
  targetPort.closest = selector => selector === '.canvas-card' ? targetCard : targetPort;
  h.setDropTarget(targetPort);
  await h.viewportHandlers.pointerdown({ target: sourcePort, preventDefault() {}, stopPropagation() {} });
  assert.equal(vm.runInContext('!!_draggingConn', h.context), true);
}

test('fresh cards render stable source and destination identities', () => {
  const h = harness();
  setup(h, current, current);
  assert.equal(vm.runInContext('_cards[0].el.querySelectorAll(".canvas-port.out")[1].dataset.port', h.context), 'navigation_lidar');
  vm.runInContext(`const input = _buildCardEl({id:'destination',mcpId:'g1',toolName:'target',x:0,y:0,topicIn:[{format:'sensor/pointcloud',port:'lidar'}]})`, h.context);
  assert.equal(vm.runInContext('input.querySelectorAll(".canvas-port.in")[0].dataset.port', h.context), 'lidar');
});

test('new connection persists source and destination identities', async () => {
  const h = harness();
  setup(h, current, current);
  await beginWire(h);
  h.handlers.pointerup({clientX:0, clientY:0});
  assert.equal(vm.runInContext('_connections.length', h.context), 1);
  assert.equal(vm.runInContext('_connections[0].fromPort', h.context), 'navigation_lidar');
  assert.equal(vm.runInContext('_connections[0].toPort', h.context), 'lidar');
});

for (const state of ['running', 'unknown']) {
  test(`pointerup refuses a connection when project state becomes ${state}`, async () => {
    const h = harness();
    setup(h, current, current);
    await beginWire(h);
    vm.runInContext(state === 'running' ? '_projectRunning = true' : '_projectStateKnown = false', h.context);
    h.handlers.pointerup({clientX:0, clientY:0});
    assert.equal(vm.runInContext('_connections.length', h.context), 0);
    assert.equal(vm.runInContext('_draggingConn', h.context), null);
  });
}
