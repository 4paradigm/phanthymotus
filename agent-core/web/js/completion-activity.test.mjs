import test from 'node:test';
import assert from 'node:assert/strict';
import {
  activityCompletionId, claimCompletionActivity, preferManualCompletionActivity,
} from './completion-activity.js';

const event = (id, status = 'completed', prefix = '') => ({type: 'trigger', payload: {
  text: prefix + JSON.stringify({type: 'action_complete', action_id: id, status})}});
const collectorEvent = (id, status = 'completed', extra = {}) => ({
  type: 'trigger', mcp_id: 'collector', payload: {
    text: '<event source="acp:mcp-piper" channel="sensor" ts="2026-10-16 12:04:40">\n'
      + JSON.stringify({type: 'action_complete', action_id: id, status, silent: false, ...extra})
      + '\n</event>',
  },
});

for (const first of ['canvas', 'collector']) {
  test(`${first} first: one activity row per action, unrelated actions remain visible`, () => {
    const id = `move-${first}`;
    const collectorId = activityCompletionId(event(id, 'completed', '[collector] '));
    const ids = first === 'canvas' ? [id, collectorId] : [collectorId, id];
    assert.deepEqual(ids.map(id => claimCompletionActivity(id)), [true, false]);
    assert.equal(claimCompletionActivity(id + '-next'), true);
  });
}

test('manual completion replaces an early collector row and rejects late receipts', () => {
  const id = activityCompletionId(collectorEvent('manual-collector-first'));
  assert.equal(claimCompletionActivity(id, 'collector'), true);
  assert.equal(preferManualCompletionActivity(id), true);
  assert.equal(claimCompletionActivity(id), true);
  assert.equal(claimCompletionActivity(id, 'collector'), false);
  assert.equal(preferManualCompletionActivity(id), false);
  assert.equal(claimCompletionActivity(id), false);
});

test('manual response first always reserves its detailed completion row', () => {
  const id = 'manual-response-first';
  assert.equal(preferManualCompletionActivity(id), false);
  assert.equal(claimCompletionActivity(id, 'collector'), false);
  assert.equal(claimCompletionActivity(id), true);
  assert.equal(claimCompletionActivity(id, 'collector'), false);
});

test('unowned collector actions remain visible exactly once', () => {
  const id = activityCompletionId(collectorEvent('llm-only'));
  assert.equal(claimCompletionActivity(id, 'collector'), true);
  assert.equal(claimCompletionActivity(id, 'collector'), false);
});

test('a manual completion remains visible without intelligent control or a collector event', () => {
  const id = 'manual-control-off';
  assert.equal(preferManualCompletionActivity(id), false);
  assert.equal(claimCompletionActivity(id), true);
  assert.equal(claimCompletionActivity(id), false);
});

test('actual single collector envelopes correlate all terminal states', () => {
  for (const status of ['completed', 'error', 'failed', 'cancelled']) {
    assert.equal(activityCompletionId(collectorEvent(`xml-${status}`, status)), `xml-${status}`);
  }
  assert.equal(activityCompletionId(collectorEvent('xml-running', 'running')), null);
  assert.equal(activityCompletionId(collectorEvent('xml-text', 'failed', {
    result: {error: 'The literal </event> marker is part of an error string.'},
  })), 'xml-text');
  const wrapped = collectorEvent('xml-whitespace');
  wrapped.payload.text = `  \n${wrapped.payload.text}\n  `;
  assert.equal(activityCompletionId(wrapped), 'xml-whitespace');
});

test('collector batches and unrelated or malformed envelopes remain visible', () => {
  const single = collectorEvent('xml-batch-a').payload.text;
  const other = collectorEvent('xml-batch-b').payload.text;
  const unrelated = '<event source="voice" channel="sensor" ts="now">\nHello\n</event>';
  for (const text of [
    `${single}\n${other}`, `${single}\n${unrelated}`, `${unrelated}\n${single}`,
    `History: ${single}`, `${single}\nAdditional information`,
    '<event source="acp">{broken}</event>',
    '<event source="voice">{"text":"Hello"}</event>',
  ]) {
    assert.equal(activityCompletionId({type: 'trigger', payload: {text}}), null);
  }
});

test('empty ownership IDs do not reserve or suppress status query rows', () => {
  for (const id of ['', null, undefined]) {
    assert.equal(preferManualCompletionActivity(id), false);
    assert.equal(claimCompletionActivity(id, 'collector'), true);
  }
});

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
