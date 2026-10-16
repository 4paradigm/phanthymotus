import test from 'node:test';
import assert from 'node:assert/strict';
import {piperDevices, switchPiperDevice} from './piper-selector.js';

const mcps = [2, 1].map(n => ({id: `mcp${n}`, name: `机械臂 ${n}`,
  tools: [{name: 'piper_joint'}, {name: 'piper_photo'}]}));
const devices = piperDevices(mcps);
const good = {busy: false, feedback_fresh: true, motion_enabled: false};

test('explicit device names preserve MCP identity independently of list order', () => {
  assert.deepEqual(devices.map(d => [d.id, d.label]), [['mcp1', '机械臂 1'], ['mcp2', '机械臂 2']]);
  assert.deepEqual(piperDevices([mcps[0]]), []);
  assert.deepEqual(piperDevices([mcps[0], {...mcps[1], name: '机械臂 2'}]), []);
});

test('switch closes both permissions and rechecks before selecting; never sends physical actions', async () => {
  const calls = [];
  const target = await switchPiperDevice(devices, 'mcp2', {projectStopped: async () => true,
    call: async (id, tool, args) => {calls.push([id, tool, args.action]);return good;}});
  assert.equal(target, 'mcp2');
  assert.deepEqual(calls.filter(c => c[1] === 'piper_permission'), [
    ['mcp1', 'piper_permission', 'block'], ['mcp2', 'piper_permission', 'block'],
  ]);
  assert.equal(calls.filter(c => c[2] === 'read').length, 4);
  assert.ok(calls.every(c => ['piper_status', 'piper_permission'].includes(c[1])));
});

test('busy, stale or offline device prevents switching before changing permissions', async () => {
  for (const state of [{...good, busy: true}, {...good, feedback_fresh: false}, {}]) {
    let blocks = 0;
    await assert.rejects(switchPiperDevice(devices, 'mcp2', {projectStopped: async () => true,
      call: async (id, tool) => {if (tool === 'piper_permission') blocks++;return state;}}));
    assert.equal(blocks, 0);
  }
  await assert.rejects(switchPiperDevice(devices, 'mcp2', {projectStopped: async () => true,
    call: async () => {throw Error('offline');}}), /offline/);
});

test('failed permission closure or state change never returns a new target', async () => {
  for (const failure of [{motion_enabled: true}, {motion_enabled: false, operation_stopping: true}]) {
    await assert.rejects(switchPiperDevice(devices, 'mcp2', {projectStopped: async () => true,
      call: async (id, tool) => tool === 'piper_permission' ? failure : good}));
  }
  let reads = 0;
  await assert.rejects(switchPiperDevice(devices, 'mcp2', {projectStopped: async () => true,
    call: async (id, tool, args) => args.action === 'read' && ++reads > 2 ? {...good, motion_enabled: true} : good}));
});

test('running project or unknown target prevents any driver call', async () => {
  for (const [target, stopped] of [['mcp2', false], ['missing', true]]) {
    let calls = 0;
    await assert.rejects(switchPiperDevice(devices, target, {projectStopped: async () => stopped,
      call: async () => {calls++;}}));
    assert.equal(calls, 0);
  }
});
