// Exercise the actual card control without importing browser app startup.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const source = fs.readFileSync(path.join(__dirname, '../web/js/sidebar.js'), 'utf8');
const helper = source.slice(source.indexOf('function _bridgePrinterControls('), source.indexOf('export async function openToolConfigModal('));
const reader = source.slice(source.indexOf('function _readConfigValue('), source.indexOf('/** Check if a configSchema has any instance-scope fields. */'));

class Element {
  constructor() { this.children = []; this.dataset = {}; this.listeners = {}; this.value = ''; this.isConnected = true; }
  add(option) { this.children.push(option); if (this.children.length === 1) this.value = option.value; }
  replaceChildren(...options) { this.children = []; this.value = ''; options.forEach(option => this.add(option)); }
  setAttribute() {}
  addEventListener(event, handler) { this.listeners[event] = handler; }
  removeEventListener(event) { delete this.listeners[event]; }
}

function fixture(fetch) {
  const fields = {};
  const initial = {bridge_url: '10.100.10.14', remote_start_user: 'operator',
    remote_start_password: 'secret', remote_start_port: '22', visitor_page_url: ''};
  for (const [key, value] of Object.entries(initial)) { fields[key] = Object.assign(new Element(), {value, dataset: {key}}); }
  const props = Object.fromEntries(Object.keys(fields).map(key => [key, {scope: 'shared', type: key === 'remote_start_enabled' ? 'boolean' : 'string'}]));
  props.visitor_page_url = {scope: 'instance'};
  props.printer_name = {scope: 'instance'};
  const body = new Element();
  body.querySelector = selector => fields[selector.match(/data-key="([^"]+)"/)[1]];
  body.querySelectorAll = () => Object.values(fields);
  const jobs = [];
  const polls = [];
  const context = vm.createContext({document: {createElement: () => new Element()},
    Option: function(text, value) { this.text = text; this.value = value; },
    fetch, ensureEdit: async () => true, _toolConfigs: {},
    queueMicrotask: fn => jobs.push(fn), setTimeout: (fn, delay) => { (delay === 2500 ? polls : jobs).push(fn); return jobs.length; }, clearTimeout: () => {}});
  vm.runInContext(reader + helper, context);
  const controls = context._bridgePrinterControls({mcpId: 'perception', toolName: 'Visitor_Badge_Printer',
    bodyEl: body, overlay: {classList: {contains: () => false}}, props, key: 'printer_name', saved: 'Brother'});
  fields.printer_name = controls.input;
  const flush = async () => { while (jobs.length) { jobs.shift()(); await new Promise(resolve => setImmediate(resolve)); } };
  return {fields, body, controls, flush, context, polls};
}

const response = result => ({ok: true, json: async () => ({code: 200, data: result})});

for (const toolName of ['Visitor_Badge_Printer', 'visual_depth']) {
  test(`canvas gear opens the complete ${toolName} configuration dialog`, async () => {
    const nodes = {};
    const makeNode = () => {
      const node = new Element();
      node.style = {};
      node.appendChild = child => node.children.push(child);
      node.querySelector = () => null;
      node.cloneNode = () => makeNode();
      node.parentNode = {replaceChild() {}};
      return node;
    };
    const hidden = new Set(['hidden']);
    nodes['tool-config-overlay'] = makeNode();
    nodes['tool-config-overlay'].classList = {
      add: value => hidden.add(value), remove: value => hidden.delete(value),
    };
    const context = vm.createContext({
      document: {createElement: makeNode, getElementById: id => nodes[id] ||= makeNode()},
      isProjectRunning: () => false, ensureEdit: async () => true,
      _toolConfigs: {}, _configEntries: props => Object.entries(props),
    });
    const start = source.indexOf('export async function openInstanceConfigModal(');
    const end = source.indexOf('// ── Helpers', start);
    vm.runInContext(source.slice(start, end).replace('export async', 'async'), context);
    await context.openInstanceConfigModal('perception', toolName, 'card-1', {properties: {}});
    assert.equal(hidden.has('hidden'), false);
    assert.equal(nodes['tool-config-title'].textContent, `Instance Config: ${toolName}`);
  });
}

test('automatic refresh discovers new pages and stops when dialog is removed', async () => {
  let pages = [];
  let calls = 0;
  const f = fixture(async () => { calls++; return response({printers: [{name: 'Brother'}], pages}); });
  await f.flush();
  pages = [{title: 'Visitors', url: 'https://visitor.test/list'}];
  await f.polls.shift()();
  assert.ok(f.fields.visitor_page_url.children.some(p => p.value === pages[0].url));
  assert.equal(f.controls.input.value, 'Brother');
  f.controls.controls[0].isConnected = false;
  const before = calls;
  await f.polls.shift()();
  assert.equal(calls, before);
  assert.equal(f.polls.length, 0);
});

test('initialization verifies SSH and persists connection before printer selection', async () => {
  const requests = [];
  const f = fixture(async (url, opts) => {
    const payload = JSON.parse(opts.body); requests.push({url, payload});
    if (opts.method === 'PUT') return {ok: true};
    const args = payload.arguments;
    if (args.action === 'initialize') return response({status: 'installing', deployment_id: 'job-1'});
    if (args.action === 'initialize_status') return response({status: 'installed', deployment_id: 'job-1', bridge_url: 'ssh://10.100.10.14'});
    return response({printers: [{name: 'Brother'}], pages: []});
  });
  const install = f.controls.controls[0].onclick();
  await new Promise(resolve => setImmediate(resolve));
  await f.flush(); await install;
  assert.equal(requests.find(r => r.payload.arguments?.action === 'initialize').payload.arguments.token, undefined);
  assert.equal(requests.find(r => r.payload.arguments?.action === 'initialize_status').payload.arguments.deployment_id, 'job-1');
  const saved = requests.find(r => !r.payload.arguments).payload;
  assert.equal(saved.token, undefined);
  assert.equal(saved.bridge_url, '10.100.10.14');
  assert.equal(saved.printer_name, undefined);
  assert.equal(f.controls.input.value, 'Brother');
});

test('printer lookup failure retains the previously selected printer', async () => {
  const f = fixture(async () => { throw new Error('network unavailable'); });
  await f.flush();
  assert.equal(f.controls.input.value, 'Brother');
  assert.equal(f.controls.input.disabled, false);
});

test('changing computer while initialization is pending cannot persist old credentials', async () => {
  let finish;
  const writes = [];
  const f = fixture(async (url, opts) => {
    if (opts.method === 'PUT') writes.push(opts);
    return new Promise(resolve => { finish = resolve; });
  });
  const install = f.controls.controls[0].onclick();
  await new Promise(resolve => setImmediate(resolve));
  f.fields.bridge_url.value = '10.100.10.20';
  finish(response({status: 'installed'}));
  await install;
  assert.equal(writes.length, 0);
});

test('stale printer lookup cannot overwrite the new computer selection', async () => {
  let finish;
  const f = fixture(async () => new Promise(resolve => { finish = resolve; }));
  await f.flush();
  f.fields.bridge_url.value = '10.100.10.20';
  f.body.listeners.input({target: f.fields.bridge_url});
  finish(response({printers: [{name: 'Old computer printer'}]}));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.controls.input.value, '');
  assert.equal(f.controls.input.children.some(option => option.value === 'Old computer printer'), false);
});

test('mismatched installation job never saves another job result', async () => {
  const writes = [];
  const f = fixture(async (url, opts) => {
    if (opts.method === 'PUT') { writes.push(opts); return {ok: true}; }
    const args = JSON.parse(opts.body).arguments;
    if (args.action === 'initialize') return response({status: 'installing', deployment_id: 'expected-job'});
    return response({status: 'installed', deployment_id: 'other-job'});
  });
  const install = f.controls.controls[0].onclick();
  await new Promise(resolve => setImmediate(resolve));
  await f.flush(); await install;
  assert.equal(writes.length, 0);
  assert.match(f.controls.controls[1].textContent, /初始化结果不匹配/);
});

test('editing address refreshes printers through the new SSH target', async () => {
  const requests = [];
  const f = fixture(async (url, opts) => { requests.push(JSON.parse(opts.body).arguments); return response({printers: []}); });
  await f.flush();
  requests.length = 0;
  f.fields.bridge_url.value = '10.100.10.99';
  f.body.listeners.input({target: f.fields.bridge_url});
  await f.flush();
  assert.equal(requests.some(args => args?.action === 'printers' && args.bridge_url === '10.100.10.99'), true);
});
