import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const here = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(process.env.CANVAS_JS || path.join(here, '../web/js/canvas.js'), 'utf8');
const render = source.slice(source.indexOf('function _buildCardEl('), source.indexOf('\nfunction _fmtColorClass(', source.indexOf('function _buildCardEl(')));
const execute = source.slice(source.indexOf('async function _executeCard('), source.indexOf('\nfunction _truncate(', source.indexOf('async function _executeCard(')));

function fixture(type, schema, running = true) {
  const calls = [];
  class Element {
    constructor() { this.dataset = {}; this.style = {}; this.innerHTML = ''; this.nodes = new Map(); }
    addEventListener(name, fn) { this[name] = fn; }
    querySelectorAll() { return []; }
    querySelector(selector) {
      if (selector === '.canvas-exec-btn' && !this.innerHTML.includes('class="canvas-exec-btn')) return null;
      if (selector === '.canvas-view-btn' && !this.innerHTML.includes('canvas-view-btn')) return null;
      if (selector === '.canvas-card-instance-cfg-btn') return null;
      if (!this.nodes.has(selector)) this.nodes.set(selector, new Element());
      return this.nodes.get(selector);
    }
  }
  const ctx = vm.createContext({
    document: { createElement: () => new Element() },
    _allMcps: [{id: 'device', tools: [{name: 'follow', type, inputSchema: schema, topic_in: [], topic_out: []}]}],
    _projectRunning: running, _cards: [], _esc: s => String(s ?? ''),
    _fmtColorClass: () => '', _ensureEdit: async () => running,
    _logActivity() {}, _showResult() {}, _formatCallResult: JSON.stringify,
    _truncate: s => s, console,
    fetch: async (url, options) => {
      calls.push({url, body: JSON.parse(options.body)});
      return {json: async () => ({code: 200, data: {accepted: true}})};
    },
  });
  vm.runInContext(render + '\n' + execute, ctx);
  const el = ctx._buildCardEl({id: 'card', mcpId: 'device', toolName: 'follow', driverName: 'driver', x: 0, y: 0});
  return {el, calls};
}

const empty = {type: 'object', properties: {}, required: [], additionalProperties: false};

test('zero-argument actuator renders Execute, no fields, and sends empty arguments', async () => {
  const {el, calls} = fixture('actuator', empty);
  assert.ok(el.innerHTML.includes('class="canvas-exec-btn'));
  assert.ok(!el.innerHTML.includes('<select'));
  assert.ok(!el.innerHTML.includes('canvas-field-input'));
  const button = el.querySelector('.canvas-exec-btn');
  await button.click({stopPropagation() {}});
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].body, {tool: 'follow', arguments: {}});
  assert.equal(button.disabled, false);
});

test('zero-argument actuator stays locked while the project is stopped', async () => {
  const {el, calls} = fixture('actuator', empty, false);
  assert.ok(el.innerHTML.includes('canvas-exec-btn locked'));
  await el.querySelector('.canvas-exec-btn').click({stopPropagation() {}});
  assert.equal(calls.length, 0);
});

test('lifecycle-only actuator and incomplete schema do not gain Execute', () => {
  for (const schema of [
    null, {}, {type: 'object'},
    {...empty, properties: {action: {type: 'string', enum: ['start', 'stop', 'info', 'config']}}},
    {...empty, required: ['missing']},
  ]) {
    const {el} = fixture('actuator', schema);
    assert.ok(!el.innerHTML.includes('class="canvas-exec-btn'), JSON.stringify(schema));
  }
});

test('no-argument processor, sensor, and controller behavior stays unchanged', () => {
  for (const type of ['processor', 'sensor', 'controller']) {
    const {el} = fixture(type, empty);
    assert.ok(!el.innerHTML.includes('class="canvas-exec-btn'), type);
  }
});

test('existing actuator with a visible action keeps its selector and Execute', () => {
  const {el} = fixture('actuator', {...empty, properties: {action: {type: 'string', enum: ['follow', 'info']}}});
  assert.ok(el.innerHTML.includes('<select'));
  assert.ok(el.innerHTML.includes('class="canvas-exec-btn'));
});
