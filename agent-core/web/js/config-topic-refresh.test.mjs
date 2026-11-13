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
  const port = { dataset: { format: 'image/jpeg', topic: `/old/${id}`, staticTopic: `/old/${id}` } };
  const inPort = { dataset: { topic: '' } };
  return { id, mcpId, toolName, topicOut: [{ topic: `/old/${id}`, format: 'image/jpeg' }],
    topicOutFrom: inputKey([]), _topicFetchFor: inputKey([]),
    el: {
      querySelectorAll: selector => selector === '.canvas-port.in' ? [inPort] : [port],
      querySelector: selector => selector.startsWith('.canvas-port.out') ? port
        : selector.startsWith('.canvas-port.in') ? inPort
        : selector === '.canvas-card-driver' ? {} : null,
      replaceWith() {},
    }, port, inPort };
}
function refreshHarness(cards, fetch, connections = []) {
  const document = new EventTarget();
  let changes = 0, saves = 0;
  document.addEventListener('canvas-topics-updated', () => { changes++; });
  const api = runInNewContext(
    code('_parseMcpCallResult') + code('_fetchTopicsFromDriver') + code('_refreshConfiguredTopics')
    + code('updateCanvasMcps') + code('_resolveAllTopics')
    + '\n({ refresh: typeof _refreshConfiguredTopics === "function" ? _refreshConfiguredTopics : undefined, info: _fetchTopicsFromDriver, poll: updateCanvasMcps });', {
    _cards: cards, fetch, inputArgs, inputKey, document, Event, console,
    _allMcps: [], _draggingCardId: null, _connections: connections,
    _inputTopicsFor: () => [], _revalidateDerivedTopics() {}, _scheduleRedraw() {},
    _buildCardEl: c => cards.find(old => old.id === c.id).el,
    _makeDraggable() {}, _syncGeometryObservers() {},
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

test('state-only info preserves output bindings without marking static topics as verified', async () => {
  const c = card('a');
  delete c.topicOutFrom;
  const p = refreshHarness([c], async () => ({ json: async () => ({ code: 200, data: { state: 'running' } }) }));
  await p.info(c, [], { force: true });
  assert.equal(c.topicOut[0].topic, '/old/a');
  assert.equal(c.port.dataset.topic, '/old/a');
  assert.equal(c.topicOutFrom, undefined);
  assert.equal(p.saves(), 0);
  assert.equal(p.changes(), 0);
});

function staticMetadata() {
  return [{ id: 'robot', tools: [{ name: 'sensor', type: 'sensor', multiInstance: false,
    topic_out: [{ topic: '/old/a', format: 'image/jpeg' }] }] }];
}

test('metadata poll cannot undo config-dependent info topics or downstream bindings', async () => {
  const source = card('a'), downstream = card('b', 'robot', 'asr');
  const edge = { fromCardId: 'a', fromPortIdx: 0, toCardId: 'b', toPortIdx: 0 };
  const p = refreshHarness([source, downstream], async () => reply('/new/output'), [edge]);
  await p.info(source, [], { force: true });
  p.poll(staticMetadata());
  assert.equal(source.topicOut[0].topic, '/new/output');
  assert.equal(source.port.dataset.topic, '/new/output');
  assert.equal(downstream.inPort.dataset.topic, '/new/output');
  assert.equal(edge.fromTopic, '/new/output');
});

test('explicit empty outputs clear bindings and stay empty through metadata polling', async () => {
  const c = card('a');
  const p = refreshHarness([c], async () => ({ json: async () => ({ code: 200, data: { topic_out: [] } }) }));
  await p.info(c, [], { force: true });
  p.poll(staticMetadata());
  assert.equal(c.topicOut.length, 0);
  assert.equal(c.port.dataset.topic, '');
  assert.equal(p.changes(), 1);
});

test('unverified static declarations still follow the metadata poll', () => {
  const c = card('a');
  delete c.topicOutFrom;
  const p = refreshHarness([c], async () => reply('/unused'));
  const metadata = staticMetadata();
  metadata[0].tools[0].topic_out[0].topic = '/metadata/new';
  p.poll(metadata);
  assert.equal(c.topicOut[0].topic, '/metadata/new');
  assert.equal(c.port.dataset.topic, '/metadata/new');
});
