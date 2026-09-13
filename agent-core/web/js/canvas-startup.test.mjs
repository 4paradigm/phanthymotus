import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

// Execute the real startup event handler, with only its browser/network edges
// replaced. This catches duplicate-name rows and premature loading completion.
const source = readFileSync(new URL('./canvas.js', import.meta.url), 'utf8');
const start = source.indexOf('async function _startProject()');
const end = source.indexOf('\nfunction _stopProject()', start);
assert(start >= 0 && end > start);
const startup = source.slice(start, end).replace(
  "await import('./motus-stream.js')", 'eventBus');

async function replay(cards, events) {
  let handler;
  const state = { rows: [], countdowns: 0, waiting: 0 };
  const context = {
    eventBus: {
      onMotusEvent: (_, fn) => { handler = fn; },
      offMotusEvent: () => {},
    },
    _saveLayout: async () => {}, _cards: [], _projectRunning: false,
    _syncProjectBtn: () => {}, _logActivity: () => {},
    document: { querySelectorAll: () => [] },
    _showStartupModal: items => {
      state.rows = Array.from(items, ({ card }) => ({ id: card.id, status: 'waiting' }));
      return {
        updateItem: (i, status) => { state.rows[i].status = status; },
        startCountdown: () => { state.countdowns++; },
        setWaiting: n => { state.waiting = n; },
      };
    },
    fetch: async () => {
      handler({ type: 'project_start_begin', payload: { cards } });
      for (const event of events) handler(event);
      return { ok: true };
    },
  };
  await vm.runInNewContext(startup + '\n_startProject()', context);
  return state;
}

const cameras = ['rgb-card', 'depth-card', 'ir-card'].map(instance_id => ({
  tool: 'ext_camera', mcp_id: 'go2', instance_id,
}));
const item = (card, status) => ({ type: 'project_start_item', payload: { ...card, status } });
const done = { type: 'project_start_done', payload: { has_error: false } };

test('all same-name camera rows reach ready', async () => {
  const result = await replay(cameras, [
    ...cameras.flatMap(card => [item(card, 'starting'), item(card, 'ready')]), done,
  ]);
  assert.deepEqual(result.rows.map(r => r.status), ['ready', 'ready', 'ready']);
  assert.deepEqual(result.rows.map(r => r.id), cameras.map(c => c.instance_id));
  assert.equal(result.countdowns, 1);
});

test('one settled instance does not hide another instance still loading', async () => {
  const result = await replay(cameras, [
    item(cameras[0], 'ready'), item(cameras[1], 'loading'), item(cameras[2], 'loading'),
    done, item(cameras[2], 'ready'),
  ]);
  assert.deepEqual(result.rows.map(r => r.status), ['ready', 'loading', 'ready']);
  assert.equal(result.countdowns, 0);
});

test('errors and cancellation settle only their own rows', async () => {
  const result = await replay(cameras, [
    ...cameras.map(card => item(card, 'loading')), done,
    item(cameras[2], 'cancelled'), item(cameras[0], 'ready'), item(cameras[1], 'error'),
  ]);
  assert.deepEqual(result.rows.map(r => r.status), ['ready', 'error', 'cancelled']);
  assert.equal(result.countdowns, 1);
});

test('legacy single-card events without instance_id remain supported', async () => {
  const legacy = { tool: 'mic', mcp_id: 'go2' };
  const result = await replay([legacy], [item(legacy, 'ready'), done]);
  assert.equal(result.rows[0].status, 'ready');
  assert.equal(result.countdowns, 1);
});

test('an event for another instance cannot keep this modal waiting', async () => {
  const result = await replay(cameras, [
    ...cameras.map(card => item(card, 'ready')),
    item({ ...cameras[0], instance_id: 'deleted-card' }, 'loading'), done,
  ]);
  assert.equal(result.countdowns, 1);
});
