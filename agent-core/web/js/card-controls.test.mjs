import test from 'node:test';
import assert from 'node:assert/strict';
import { fieldOptions, confirmedAction, confirmationText } from './card-controls.js';

const schema = {'x-confirmation': {move: {
  preview: {action: 'move', execute: false},
  execute: {execute: true, workspace_clear: true, take_control: true},
  message: '确认现场条件',
}}};

test('Chinese labels preserve exact protocol values and booleans', () => {
  assert.deepEqual(fieldOptions({enum: [false, true], oneOf: [
    {const: false, title: '未确认'}, {const: true, title: '已确认'},
  ]}), [{value: false, label: '未确认'}, {value: true, label: '已确认'}]);
});

test('preview happens first; operator confirmation carries its token and unchanged target', async () => {
  const calls = [], displayed = [];
  const result = await confirmedAction(schema, {action: 'move', joint: 1, target_deg: 2}, {
    call: async args => {calls.push(args);return args.execute ? {executed: true} : {preview_id: 'fresh', target_deg: [2]};},
    confirm: async text => {assert.match(text, /目标角度/);return true;},
    showPreview: p => displayed.push(p),
  });
  assert.equal(calls.length, 2);
  assert.equal(calls[0].execute, false);
  assert.deepEqual(calls[1], {action: 'move', joint: 1, target_deg: 2, execute: true,
    workspace_clear: true, take_control: true, preview_id: 'fresh'});
  assert.equal(displayed.length, 1);
  assert.equal(result.executed, true);
});

test('cancel and blocked/missing previews never issue execution', async () => {
  let calls = 0;
  const r = await confirmedAction(schema, {action: 'move'}, {
    call: async () => {calls++;return {preview_id: 'x'};}, confirm: () => false,
  });
  assert.equal(calls, 1);
  assert.equal(r.cancelled, true);
  for (const plan of [{allowed: false, message: '六轴仍使能'}, {}]) {
    calls = 0;
    await assert.rejects(confirmedAction(schema, {action: 'move'}, {
      call: async () => {calls++;return plan;}, confirm: () => {throw Error('must not confirm');},
    }));
    assert.equal(calls, 1);
  }
});

test('already-ready result does not re-enable or ask for needless confirmation', async () => {
  let calls = 0;
  const result = await confirmedAction(schema, {action: 'move'}, {
    call: async () => {calls++;return {confirmed: true, executed: false, message: '已就绪'};},
    confirm: () => {throw Error('must not confirm');},
  });
  assert.equal(calls, 1);
  assert.equal(result.executed, false);
});

test('capture asks explicit permission before the sole camera call', async () => {
  const events = [];
  await confirmedAction({'x-confirmation': {capture: {execute: {capture_confirmed: true}, message: '拍摄一张'}}},
    {action: 'capture'}, {confirm: () => {events.push('confirm');return true;},
      call: args => {events.push('capture');assert.equal(args.capture_confirmed, true);return {};}});
  assert.deepEqual(events, ['confirm', 'capture']);
});

test('boundary correction is disclosed; drivers without metadata keep direct execution', async () => {
  assert.match(confirmationText({message: '使能'}, {boundary_adjustment_deg: [0, 2, -1, 0, 0, 0]}), /修正角度.*0，2，-1/);
  let calls = 0;
  await confirmedAction({}, {action: 'legacy'}, {call: args => {calls++;assert.equal(args.action, 'legacy');},
    confirm: () => {throw Error('legacy must not get a new prompt');}});
  assert.equal(calls, 1);
});
