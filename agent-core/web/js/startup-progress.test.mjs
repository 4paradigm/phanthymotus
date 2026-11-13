import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

// Execute the canvas's actual event handler without mounting the whole editor.
const canvas = readFileSync(new URL('./canvas.js', import.meta.url), 'utf8');
const start = canvas.indexOf('  // Subscribe to startup progress events');
const end = canvas.indexOf('  onMotusEvent(null, _onEvent);', start);
assert.ok(start >= 0 && end > start);

const cards = ['left', 'right'].map(card_id => ({
  card_id, mcp_id: 'tianyi', tool: 'wrist_camera',
}));

function progress(canvasCards = []) {
  const rows = [];
  const countdowns = [];
  const waiting = [];
  let detached = false;
  let modalCards;
  const onEvent = runInNewContext(canvas.slice(start, end) + '\n_onEvent;', {
    _cards: canvasCards,
    _showStartupModal(items) {
      modalCards = items.map(i => i.card);
      rows.push(...items.map(() => ({ status: 'waiting' })));
      return {
        updateItem(i, status, message) { rows[i] = { status, message }; },
        startCountdown(seconds) { countdowns.push(seconds); },
        setWaiting(count) { waiting.push(count); },
      };
    },
    offMotusEvent() { detached = true; },
  });
  onEvent({ type: 'project_start_begin', payload: { cards } });
  return {
    rows, countdowns, waiting, detached: () => detached, cards: () => modalCards,
    item(id, status, message = '') {
      onEvent({ type: 'project_start_item', payload: {
        card_id: id, mcp_id: 'tianyi', tool: 'wrist_camera', status, message,
      } });
    },
    done(has_error = false) {
      onEvent({ type: 'project_start_done', payload: { has_error } });
    },
  };
}

test('startup rows retain the corresponding canvas card identity', () => {
  const p = progress([
    { id: 'right', mcpId: 'tianyi', toolName: 'wrist_camera', driverName: 'Robot' },
    { id: 'left', mcpId: 'tianyi', toolName: 'wrist_camera', driverName: 'Robot' },
  ]);
  assert.deepEqual(Array.from(p.cards(), c => c.id), ['left', 'right']);
  const missing = progress();
  assert.deepEqual(Array.from(missing.cards(), c => c.id), ['left', 'right']);
});

// Minimal DOM surface for executing the real row-rendering code.
class Element {
  constructor() { this.children = []; this.nodes = {}; this.style = {}; }
  appendChild(child) { this.children.push(child); }
  querySelector(selector) {
    if (!this.nodes[selector]) {
      this.nodes[selector] = new Element();
      if (selector === '.startup-name') {
        this.nodes[selector].textContent = this.innerHTML?.match(/class="startup-name">([^<]*)</)?.[1] || '';
      }
    }
    return this.nodes[selector];
  }
  addEventListener() {}
}

function modalNames(inputCards, configs = {}) {
  const at = canvas.indexOf('function _showStartupModal(');
  const end = canvas.indexOf('\n}\n', at) + 3;
  const show = runInNewContext(canvas.slice(at, end) + '\n_showStartupModal;', {
    document: { createElement: () => new Element(), body: new Element() },
    getToolConfigs: () => configs,
  });
  const result = show(inputCards.map(card => ({ card })));
  return result.modal.querySelector('.startup-modal-list').children.map(row =>
    row.querySelector('.startup-name').textContent);
}

test('same-tool rows show configured wrist side and unique instance ID', () => {
  const input = [
    { id: 'card-a', toolName: 'wrist_camera', mcpId: 'tianyi' },
    { id: 'card-b', toolName: 'wrist_camera', mcpId: 'tianyi' },
  ];
  assert.deepEqual(modalNames(input, {
    'tianyi:wrist_camera:card-a': { side: 'left' },
    'tianyi:wrist_camera:card-b': { side: 'right' },
  }), ['wrist_camera · 左侧 · card-a', 'wrist_camera · 右侧 · card-b']);
});

test('same-side or unlabelled instances remain distinguishable by ID', () => {
  const input = ['a', 'b'].map(id => ({ id, toolName: 'sensor', mcpId: 'robot' }));
  assert.deepEqual(modalNames(input), ['sensor · a', 'sensor · b']);
  assert.deepEqual(modalNames(input, {
    'robot:sensor:a': { side: 'left' }, 'robot:sensor:b': { side: 'left' },
  }), ['sensor · 左侧 · a', 'sensor · 左侧 · b']);
});

test('single-instance cards keep the existing tool name', () => {
  assert.deepEqual(modalNames([{ id: 'head', toolName: 'camera_head', mcpId: 'robot' }]),
    ['camera_head']);
});

test('two instances of one tool update their own modal rows', () => {
  const p = progress();
  p.item('left', 'starting');
  assert.equal(p.rows[0].status, 'starting');
  assert.equal(p.rows[1].status, 'waiting');
  p.item('left', 'ready');
  p.item('right', 'ready');
  p.done();
  assert.deepEqual(p.rows.map(r => r.status), ['ready', 'ready']);
  assert.deepEqual(p.countdowns, [15]);
});

test('one ready instance cannot clear another instance still loading', () => {
  const p = progress();
  p.item('left', 'loading');
  p.item('right', 'loading');
  p.item('left', 'ready');
  p.done();
  assert.deepEqual(p.rows.map(r => r.status), ['ready', 'loading']);
  assert.deepEqual(p.waiting, [1]);
  assert.deepEqual(p.countdowns, []);
  assert.equal(p.detached(), false);
  p.item('right', 'ready');
  assert.deepEqual(p.countdowns, [15]);
  assert.equal(p.detached(), true);
});

for (const status of ['error', 'cancelled']) {
  test(`${status} belongs to its instance rather than the last same-tool row`, () => {
    const p = progress();
    p.item('right', 'ready');
    p.item('left', status, 'left camera outcome');
    assert.deepEqual(p.rows.map(r => r.status), [status, 'ready']);
    assert.equal(p.rows[0].message, 'left camera outcome');
  });
}
