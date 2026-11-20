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
  const initial = {bridge_url: '10.100.10.14', remote_start_user: 'operator', remote_start_port: '22',
    remote_start_key_path: '/run/ql-printer-auth/frontdesk_bridge_ed25519', token: '', remote_start_enabled: 'false'};
  for (const [key, value] of Object.entries(initial)) { fields[key] = Object.assign(new Element(), {value, dataset: {key}}); }
  const props = Object.fromEntries(Object.keys(fields).map(key => [key, {scope: 'shared', type: key === 'remote_start_enabled' ? 'boolean' : 'string'}]));
  props.printer_name = {scope: 'instance'};
  const body = new Element();
  body.querySelector = selector => fields[selector.match(/data-key="([^"]+)"/)[1]];
  body.querySelectorAll = () => Object.values(fields);
  const jobs = [];
  const context = vm.createContext({document: {createElement: () => new Element()},
    Option: function(text, value) { this.text = text; this.value = value; },
    fetch, ensureEdit: async () => true, _toolConfigs: {},
    queueMicrotask: fn => jobs.push(fn), setTimeout: fn => { jobs.push(fn); return jobs.length; }, clearTimeout: () => {}});
  vm.runInContext(reader + helper, context);
  const controls = context._bridgePrinterControls({mcpId: 'perception', toolName: 'Visitor_Badge_Printer',
    bodyEl: body, overlay: {classList: {contains: () => false}}, props, key: 'printer_name', saved: 'Brother'});
  fields.printer_name = controls.input;
  const flush = async () => { while (jobs.length) { jobs.shift()(); await new Promise(resolve => setImmediate(resolve)); } };
  return {fields, body, controls, flush, context};
}

const response = result => ({ok: true, json: async () => ({code: 200, data: result})});

test('initialization pairs automatically and persists token before printer selection', async () => {
  const requests = [];
  const f = fixture(async (url, opts) => {
    const payload = JSON.parse(opts.body); requests.push({url, payload});
    if (opts.method === 'PUT') return {ok: true};
    const args = payload.arguments;
    if (args.action === 'initialize') return response({status: 'installing', deployment_id: 'job-1'});
    if (args.action === 'initialize_status') return response({status: 'installed', deployment_id: 'job-1', token: 'generated-or-existing', bridge_url: 'http://10.100.10.14:8765'});
    return response({printers: [{name: 'Brother'}]});
  });
  const install = f.controls.controls[0].onclick();
  await new Promise(resolve => setImmediate(resolve));
  await f.flush(); await install;
  assert.equal(requests.find(r => r.payload.arguments?.action === 'initialize').payload.arguments.token, '');
  assert.equal(requests.find(r => r.payload.arguments?.action === 'initialize_status').payload.arguments.deployment_id, 'job-1');
  const saved = requests.find(r => !r.payload.arguments).payload;
  assert.equal(saved.token, 'generated-or-existing');
  assert.equal(saved.remote_start_enabled, true);
  assert.equal(saved.bridge_url, 'http://10.100.10.14:8765');
  assert.equal(saved.printer_name, undefined);
  assert.equal(f.controls.input.value, 'Brother');
});

test('printer lookup failure retains the previously selected printer', async () => {
  const f = fixture(async () => { throw new Error('network unavailable'); });
  f.fields.token.value = 'existing';
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
  finish(response({status: 'installed', token: 'old-computer-secret'}));
  await install;
  assert.equal(writes.length, 0);
  assert.equal(f.fields.token.value, '');
});

test('stale printer lookup cannot overwrite the new computer selection', async () => {
  let finish;
  const f = fixture(async () => new Promise(resolve => { finish = resolve; }));
  f.fields.token.value = 'existing';
  await f.flush();
  f.fields.bridge_url.value = '10.100.10.20';
  f.body.listeners.input({target: f.fields.bridge_url});
  assert.equal(f.fields.token.value, '');
  finish(response({printers: [{name: 'Old computer printer'}]}));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.controls.input.value, '');
  assert.equal(f.controls.input.children.some(option => option.value === 'Old computer printer'), false);
});

test('mismatched installation job never saves another job token', async () => {
  const writes = [];
  const f = fixture(async (url, opts) => {
    if (opts.method === 'PUT') { writes.push(opts); return {ok: true}; }
    const args = JSON.parse(opts.body).arguments;
    if (args.action === 'initialize') return response({status: 'installing', deployment_id: 'expected-job'});
    return response({status: 'installed', deployment_id: 'other-job', token: 'unrelated-secret'});
  });
  const install = f.controls.controls[0].onclick();
  await new Promise(resolve => setImmediate(resolve));
  await f.flush(); await install;
  assert.equal(writes.length, 0);
  assert.equal(f.fields.token.value, '');
  assert.match(f.controls.controls[1].textContent, /初始化结果不匹配/);
});

test('editing address clears credentials and does not query the new computer before pairing', async () => {
  const requests = [];
  const f = fixture(async (url, opts) => { requests.push(JSON.parse(opts.body).arguments); return response({printers: []}); });
  f.fields.token.value = 'old-computer-token';
  await f.flush();
  requests.length = 0;
  f.fields.bridge_url.value = '10.100.10.99';
  f.body.listeners.input({target: f.fields.bridge_url});
  await f.flush();
  assert.equal(f.fields.token.value, '');
  assert.equal(requests.length, 0);
});
