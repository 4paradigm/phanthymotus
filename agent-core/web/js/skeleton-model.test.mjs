import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resolveSkeletonMcp } from './renderers/skeleton-model.js';

const service = (id, topic, online = true) => ({ id, online,
  tools: [{ name: 'model', type: 'resource' }, { name: 'joint_state',
    topic_out: [{ topic, format: 'sensor/skeleton' }] }] });
const response = data => async url => {
  assert.equal(url, '/api/mcp');
  return { ok: true, json: async () => ({ data }) };
};

test('detail and dashboard placeholders resolve to the real model provider', async () => {
  for (const id of ['detail', 'dashboard', null]) {
    assert.equal(await resolveSkeletonMcp(id, '/limx_tron2/joint_state', response([
      service('other', '/other'), service('mcp-tron2', '/limx_tron2/joint_state'),
    ])), 'mcp-tron2');
  }
});

test('known MCP identity is preserved without a discovery request', async () => {
  assert.equal(await resolveSkeletonMcp('mcp-real', '/topic', () => { throw new Error('must not fetch'); }), 'mcp-real');
  assert.equal(await resolveSkeletonMcp('detail', null, () => { throw new Error('must not fetch'); }), null);
});

test('ambiguous or offline providers never pick an unrelated robot model', async () => {
  await assert.rejects(resolveSkeletonMcp('detail', '/topic', response([
    service('one', '/topic'), service('two', '/topic'),
  ])), /unique/);
  await assert.rejects(resolveSkeletonMcp('detail', '/topic', response([
    service('offline', '/topic', false), service('other', '/other'),
  ])), /unique/);
});
