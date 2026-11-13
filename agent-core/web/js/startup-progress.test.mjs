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

function progress() {
  const rows = [];
  const countdowns = [];
  const waiting = [];
  let detached = false;
  const onEvent = runInNewContext(canvas.slice(start, end) + '\n_onEvent;', {
    _showStartupModal(items) {
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
    rows, countdowns, waiting, detached: () => detached,
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
