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
// Minimal DOM surface: parse the actual renderer's ports and replace real nodes,
// rather than returning the previous element when the card is rebuilt.
class Element {
  constructor() { this.dataset = {}; this.style = {}; this.children = []; this.nodes = {}; this.ports = []; }
  set innerHTML(html) {
    this.html = html;
    this.ports = [...html.matchAll(/<div class="canvas-port (in|out)[^"]*"([^>]*)>/g)].map(m => {
      const port = new Element();
      for (const attr of m[2].matchAll(/data-([a-z-]+)="([^"]*)"/g)) {
        port.dataset[attr[1].replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = attr[2];
      }
      return port;
    });
  }
  querySelectorAll(selector) {
    if (selector === '.canvas-port.in' || selector === '.canvas-port.out') {
      return this.ports.filter(p => p.dataset.dir === selector.split('.').at(-1));
    }
    return [];
  }
  querySelector(selector) {
    if (selector.startsWith('.canvas-port.')) {
      const idx = selector.match(/data-idx="([^\"]+)"/);
      return this.querySelectorAll(selector.split('[')[0]).find(p => !idx || p.dataset.idx === idx[1]) || null;
    }
    const cls = selector.slice(1);
    return this.html?.includes(cls) ? (this.nodes[selector] ||= new Element()) : null;
  }
  addEventListener() {}
  appendChild(el) { el.parent = this; this.children.push(el); }
  replaceWith(el) { const i = this.parent?.children.indexOf(this); if (i >= 0) { this.parent.children[i] = el; el.parent = this.parent; } }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(c => c !== this); }
}

function refreshHarness(cards, fetch, connections = [], { dom = false, metadata = [], revalidate = false } = {}) {
  const document = new EventTarget();
  document.createElement = () => new Element();
  const viewport = new Element();
  let changes = 0, saves = 0, persisted;
  document.addEventListener('canvas-topics-updated', () => { changes++; });
  const scope = {
    _cards: cards, inputArgs, inputKey, document, Event, console,
    _allMcps: metadata, _draggingCardId: null, _connections: connections,
    _inputTopicsFor: () => [], _revalidateDerivedTopics() {}, _scheduleRedraw() {},
    _buildCardEl: c => cards.find(old => old.id === c.id).el,
    _makeDraggable() {}, _syncGeometryObservers() {}, _syncEmptyState() {},
    _debouncedSave() { saves++; },
    _viewport: viewport, _isEditor: true, _sessionId: 'editor', _execConnections: [],
    _zoom: 1, _tx: 0, _ty: 0, _updateEditorUI() {},
    fetch: async (url, init) => {
      if (url === '/api/canvas/layout') {
        if (init) persisted = JSON.parse(init.body);
        return { status: 200, json: async () => ({ data: persisted, editor: 'editor' }) };
      }
      return fetch(url, init);
    },
  };
  const api = runInNewContext(
    (dom ? code('_fmtColorClass') + code('_esc') + code('_buildCardEl') : '')
    + (revalidate ? code('_inputTopicsFor') + code('_revalidateDerivedTopics') : '')
    + code('_parseMcpCallResult') + code('_fetchTopicsFromDriver') + code('_refreshConfiguredTopics')
    + code('updateCanvasMcps') + code('_resolveAllTopics') + code('_addCard') + code('_saveLayout') + code('_reloadLayout')
    + `\n({ refresh: _refreshConfiguredTopics, info: _fetchTopicsFromDriver, poll: updateCanvasMcps,
      add: _addCard, save: _saveLayout, reload: _reloadLayout, resolve: _resolveAllTopics, cards: () => _cards });`, scope);
  let listener, refreshing;
  const at = source.indexOf('  onMotusEvent(null, (event) => {');
  runInNewContext(source.slice(at, source.indexOf('\n  });', at) + 6), {
    onMotusEvent: (_mcp, fn) => { listener = fn; },
    _refreshConfiguredTopics: payload => { refreshing = api.refresh(payload); },
  });
  return { ...api, changes: () => changes, saves: () => saves, persisted: () => persisted,
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

test('state-only info allows a successful retry for the same input without force', async () => {
  const c = card('a');
  let reads = 0;
  const p = refreshHarness([c], async () => ++reads === 1
    ? { json: async () => ({ code: 200, data: { state: 'loading' } }) } : reply('/new/retried'));
  await p.info(c, ['/new/input']);
  assert.equal(c.topicOut[0].topic, '/old/a');
  await p.info(c, ['/new/input']);
  assert.equal(reads, 2);
  assert.equal(c.topicOut[0].topic, '/new/retried');
});

function domHarness() {
  let output = [{ topic: '/live/rgb', format: 'image/jpeg' }, { topic: '/live/depth', format: 'image/depth-zlib' }];
  const metadata = staticMetadata();
  metadata[0].tools[0].topic_out = [...output];
  const p = refreshHarness([], async () => ({ json: async () => ({ code: output === 'offline' ? 503 : 200, data: output === null
    ? { state: 'idle' } : { topic_out: output } }) }), [], { dom: true, metadata });
  p.add({ id: 'a', mcpId: 'robot', toolName: 'sensor', x: 0, y: 0 }, false);
  return { ...p, metadata, output: value => { output = value; } };
}
const ports = card => Array.from(card.el.querySelectorAll('.canvas-port.out'));

test('actual renderer removes ports for an explicit empty output and updates format-only geometry', async () => {
  const p = domHarness();
  const c = p.cards()[0];
  await p.info(c, [], { force: true });
  assert.equal(ports(c).length, 2);
  p.output([{ format: 'audio/pcm-16k' }, { format: 'text' }]);
  await p.info(c, [], { force: true });
  assert.deepEqual(ports(c).map(port => port.dataset.format), ['audio/pcm-16k', 'text']);
  assert.ok(ports(c).every(port => port.dataset.topic === ''));
  p.output([]);
  await p.info(c, [], { force: true });
  assert.equal(ports(c).length, 0);
  p.output([{ format: 'audio/pcm-16k' }]);
  await p.info(c, [], { force: true });
  assert.deepEqual(ports(c).map(port => port.dataset.format), ['audio/pcm-16k']);
  assert.equal(ports(c)[0].dataset.topic, '');
});

for (const mode of [null, 'offline']) {
  test(`explicit empty outputs survive save/reload with ${mode === null ? 'state-only info' : 'offline driver'}`, async () => {
    const p = domHarness();
    p.output([]);
    await p.info(p.cards()[0], [], { force: true });
    await p.save();
    p.output(mode);
    await p.reload();
    await p.info(p.cards()[0], [], { force: true });
    p.poll(p.metadata);
    const c = p.cards()[0];
    assert.equal(c.topicOut.length, 0);
    assert.equal(ports(c).length, 0);
    assert.equal(p.persisted().cards[0].topicOutFrom, '', 'preserve the verified empty-input signature');
  });
}

test('format-only outputs remain authoritative when metadata forces a card rebuild', async () => {
  const p = domHarness();
  p.output([{ format: 'audio/pcm-16k' }]);
  await p.info(p.cards()[0], [], { force: true });
  p.metadata[0].tools[0].topic_in = [{ topic: '/input', format: 'text' }];
  p.poll(p.metadata);
  assert.deepEqual(ports(p.cards()[0]).map(port => port.dataset.format), ['audio/pcm-16k']);
});

test('legacy empty layouts without provenance still initialize from static declarations', async () => {
  const metadata = staticMetadata();
  const p = refreshHarness([], async () => ({ json: async () => ({ code: 503 }) }), [], { dom: true, metadata });
  p.add({ id: 'a', mcpId: 'robot', toolName: 'sensor', topicOut: [] }, false);
  assert.equal(ports(p.cards()[0]).length, 1);
  assert.equal(p.cards()[0].topicOut[0].topic, '/old/a');
  await p.save();
  assert.equal('topicOutFrom' in p.persisted().cards[0], false);
});

test('restored processors revalidate saved output provenance once per page', async () => {
  let reads = 0;
  const metadata = [{ id: 'robot', tools: [{ name: 'tts', type: 'processor', multiInstance: true,
    topic_out: [{ format: 'audio/pcm-16k' }] }] }];
  const p = refreshHarness([], async () => { reads++; return reply('/new/default'); }, [],
    { dom: true, metadata, revalidate: true });
  p.add({ id: 'a', mcpId: 'robot', toolName: 'tts', topicOut: [], topicOutFrom: '' }, false);
  p.resolve();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(reads, 1, 'persisted provenance is not proof of a query in the new page');
  assert.equal(p.cards()[0].topicOut[0].topic, '/new/default');
  p.resolve();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(reads, 1, 'successful same-input verification stays cached');
});
