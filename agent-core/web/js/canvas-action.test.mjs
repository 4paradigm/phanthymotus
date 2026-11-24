import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const source = readFileSync(new URL('./canvas.js', import.meta.url), 'utf8');
const start = source.indexOf('async function _executeCard(');
const executeCard = source.slice(start, source.indexOf('\n}\n', start) + 2);

function harness(action, mapName, confirmed = true) {
  const button = { disabled: false, textContent: '▶ 执行' };
  const input = (key, value) => ({ type: 'text', dataset: { key }, value });
  const actionField = input('action', action);
  const fields = [actionField, input('map_name', mapName), input('map_id', 'stale-id')];
  const prompts = [];
  const requests = [];
  const schema = {
    properties: {
      action: { type: 'string' },
      map_name: { type: 'string' },
      map_id: { type: 'string' },
    },
    'x-action-params': {
      select_map: { params: ['map_name'] },
      delete_map: { params: ['map_name'] },
    },
    'x-action-lifecycle': {
      actions: { delete_map: { confirm: '删除地图及其全部版本？' } },
    },
  };
  const scope = {
    _allMcps: [{ id: 'perception', tools: [{ name: 'pointcloud_slam', inputSchema: schema }] }],
    confirm: message => {
      prompts.push(message);
      return confirmed;
    },
    fetch: async (url, options) => {
      requests.push({ url, body: JSON.parse(options.body) });
      return { json: async () => ({ code: 200, data: 'ok' }) };
    },
    _logActivity() {},
    _formatCallResult: value => value,
    _showResult() {},
    _truncate: value => value,
  };
  const el = {
    querySelector: selector => selector === '.canvas-exec-btn' ? button : actionField,
    querySelectorAll: selector => selector === '.canvas-field-input' ? fields : [],
  };
  const execute = runInNewContext(executeCard + '\n_executeCard;', scope);
  return { call: () => execute(el, 'perception', 'pointcloud_slam'), button, prompts, requests };
}

test('cancelling deletion leaves the map request unsent', async () => {
  const { call, button, prompts, requests } = harness('delete_map', '2F', false);
  await call();
  assert.deepEqual(prompts, ['删除地图及其全部版本？\n2F']);
  assert.deepEqual(requests, []);
  assert.equal(button.disabled, false);
});

test('confirming deletion sends one request for the exact map shown', async () => {
  const { call, prompts, requests } = harness('delete_map', 'true');
  await Promise.all([call(), call()]);
  assert.deepEqual(prompts, ['删除地图及其全部版本？\ntrue']);
  assert.deepEqual(requests, [{
    url: '/api/mcp/perception/call',
    body: { tool: 'pointcloud_slam', arguments: { action: 'delete_map', map_name: 'true' } },
  }]);
});

test('selecting a map sends only the visible action fields with declared types', async () => {
  const { call, prompts, requests } = harness('select_map', 'false');
  await call();
  assert.deepEqual(prompts, []);
  assert.deepEqual(requests[0].body.arguments, { action: 'select_map', map_name: 'false' });
});
