import test from 'node:test';
import assert from 'node:assert/strict';
import { activityCompletionId, claimCompletionActivity } from './completion-activity.js';

const event = (id, status = 'completed', prefix = '') => ({type: 'trigger', payload: {
  text: prefix + JSON.stringify({type: 'action_complete', action_id: id, status})}});

for (const first of ['canvas', 'collector']) {
  test(`${first} first: one activity row per action, unrelated actions remain visible`, () => {
    const id = `move-${first}`;
    const collectorId = activityCompletionId(event(id, 'completed', '[collector] '));
    const ids = first === 'canvas' ? [id, collectorId] : [collectorId, id];
    assert.deepEqual(ids.map(claimCompletionActivity), [true, false]);
    assert.equal(claimCompletionActivity(id + '-next'), true);
  });
}

test('no collector needed for a manual completion; status queries without IDs are not suppressed', () => {
  assert.equal(claimCompletionActivity('manual-only'), true);
  assert.equal(claimCompletionActivity(null), true);
  assert.equal(claimCompletionActivity(null), true);
});

test('all terminal states correlate, but unrelated/batched triggers are preserved', () => {
  for (const status of ['completed', 'error', 'failed', 'cancelled']) {
    assert.equal(activityCompletionId(event('a', status)), 'a');
  }
  assert.equal(activityCompletionId(event('a', 'running')), null);
  assert.equal(activityCompletionId({type: 'mcp_result', payload: event('a').payload}), null);
  for (const text of ['hello', '{broken', event('a').payload.text + '\n' + event('b').payload.text]) {
    assert.equal(activityCompletionId({type: 'trigger', payload: {text}}), null);
  }
});

test('duplicate cache stays bounded', () => {
  assert.equal(claimCompletionActivity('evicted'), true);
  for (let i = 0; i < 2048; i++) claimCompletionActivity(`bounded-${i}`);
  assert.equal(claimCompletionActivity('evicted'), true);
  assert.equal(claimCompletionActivity('bounded-2047'), false);
});
