import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { inputArgs, inputKey } from './topic-derive.js';

const source = readFileSync(new URL('./canvas.js', import.meta.url), 'utf8');
function code(name) {
  const at = source.search(new RegExp(`(?:async )?function ${name}\\(`));
  return at < 0 ? '' : source.slice(at, source.indexOf('\n}\n', at) + 3);
}
function card(id, mcpId = 'robot', toolName = 'sensor') {
  const port = { dataset: { format: 'image/jpeg', topic: `/old/${id}` } };
  return { id, mcpId, toolName, topicOut: [{ topic: `/old/${id}`, format: 'image/jpeg' }],
    topicOutFrom: inputKey([]), _topicFetchFor: inputKey([]),
    el: { querySelectorAll: () => [port] }, port };
}
function refreshHarness(cards, fetch) {
  const document = new EventTarget();
  let changes = 0, saves = 0;
  document.addEventListener('canvas-topics-updated', () => { changes++; });
  const api = runInNewContext(
    code('_parseMcpCallResult') + code('_fetchTopicsFromDriver') + code('_refreshConfiguredTopics')
    + '\n({ refresh: typeof _refreshConfiguredTopics === "function" ? _refreshConfiguredTopics : undefined, info: _fetchTopicsFromDriver });', {
    _cards: cards, fetch, inputArgs, inputKey, document, Event, console,
    _inputTopicsFor: () => [], _resolveAllTopics() {}, _scheduleRedraw() {},
    _debouncedSave() { saves++; },
  });
  let listener, refreshing;
  const at = source.indexOf('  onMotusEvent(null, (event) => {');
  runInNewContext(source.slice(at, source.indexOf('\n  });', at) + 6), {
    onMotusEvent: (_mcp, fn) => { listener = fn; },
    _refreshConfiguredTopics: payload => { refreshing = api.refresh(payload); },
  });
  return { ...api, changes: () => changes, saves: () => saves,
    config(payload) { listener({ type: 'tool_config_updated', payload }); return refreshing; } };

}
const reply = topic => ({ json: async () => ({ code: 200,
  data: { topic_out: [{ topic, format: 'image/jpeg' }] } }) });

// Catch the save->refresh break: valid cached inputs must not suppress this read.
test('configuration update refetches every matching instance, updates ports and saves topics', async () => {
  const cards = [card('a'), card('b'), card('other-driver', 'other'), card('other-tool', 'robot', 'asr')];
  const reads = [];
  const p = refreshHarness(cards, async (_url, init) => {
    const args = JSON.parse(init.body).arguments;
    reads.push(args.instance_id);
    return reply(`/new/${args.instance_id}`);
  });
  assert.equal(typeof p.refresh, 'function');
  await p.config({ mcp_id: 'robot', tool: 'sensor', instance_id: 'a' });
  assert.deepEqual(reads.sort(), ['a', 'b']);
  assert.equal(cards[0].topicOut[0].topic, '/new/a');
  assert.equal(cards[1].port.dataset.topic, '/new/b');
  assert.equal(cards[2].topicOut[0].topic, '/old/other-driver');
  assert.equal(p.changes(), 2);
  assert.equal(p.saves(), 2);
});

test('older in-flight info cannot overwrite a post-config refresh', async () => {
  const c = card('a');
  c._topicFetchFor = null;
  const pending = [];
  const p = refreshHarness([c], () => new Promise(resolve => pending.push(resolve)));
  const old = p.info(c, []);
  const fresh = p.info(c, [], { force: true });
  assert.equal(pending.length, 2, 'config changes bypass input-topic cache');
  pending[1](reply('/new/right_rgb'));
  await fresh;
  pending[0](reply('/old/left_rgb'));
  await old;
  assert.equal(c.topicOut[0].topic, '/new/right_rgb');
  assert.equal(c.port.dataset.topic, '/new/right_rgb');
  assert.equal(p.changes(), 1);
});

test('failed info refresh preserves previously verified topics and names', async () => {
  const c = card('a');
  c._topicFetchFor = null;
  const p = refreshHarness([c], async () => ({ json: async () => ({ code: 503 }) }));
  await p.info(c, [], { force: true });
  assert.equal(c.topicOut[0].topic, '/old/a');
  assert.equal(p.changes(), 0);
});
