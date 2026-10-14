import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { mergeInputPorts } from './topic-derive.js';

// Exercise the actual drop handler with canvas geometry/persistence stubbed.
const source = readFileSync(new URL('./canvas.js', import.meta.url), 'utf8');
const addCard = source.slice(source.indexOf('function _addCard('),
  source.indexOf('async function _removeCard('));

function drop(tool, save = true) {
  const calls = [];
  const context = vm.createContext({
    _allMcps: [{ id: 'bumi', tools: [tool] }], _cards: [], mergeInputPorts,
    _buildCardEl: () => ({}), _viewport: { appendChild() {} },
    _makeDraggable() {}, _syncGeometryObservers() {}, _fetchTopicsFromDriver() {},
    _syncEmptyState() {}, _saveLayout() {},
    _startDirectionSensor: (mcp, tool, id) => calls.push([mcp, tool, 'start', { instance_id: id }]),
  });
  vm.runInContext(addCard, context);
  context._addCard({ id: 'card-1', mcpId: 'bumi', toolName: tool.name }, save);
  return calls;
}

const direction = { name: 'sound_direction', type: 'sensor',
  topic_out: [{ format: 'sensor/sound-direction-v1', topic: '/ubuntu/sound_direction' }] };

test('dropping a direction sensor starts it without starting the project', () => {
  const calls = drop(direction);
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].slice(0, 3), ['bumi', 'sound_direction', 'start']);
  assert.equal(calls[0][3].instance_id, 'card-1');
});

test('restoring a saved layout does not undo an explicit project stop', () => {
  assert.equal(drop(direction, false).length, 0);
});

test('dropping unrelated cards does not start their workers', () => {
  assert.equal(drop({ name: 'asr', type: 'processor', multiInstance: true }).length, 0);
  assert.equal(drop({ name: 'mic', type: 'sensor', topic_out: [{ format: 'audio/pcm-16k' }] }).length, 0);
});

const startDirection = source.slice(source.indexOf('async function _startDirectionSensor('),
  source.indexOf('async function _triggerAction('));
const parseReply = source.slice(source.indexOf('function _parseMcpCallResult('),
  source.indexOf('/**\n * Ask the driver'));
for (const state of ['running', 'unavailable', 'error']) {
  test(`direction start handles framework reply: ${state}`, async () => {
    const messages = [];
    const context = vm.createContext({
      _triggerAction: async () => ({ code: 200, data: [{ type: 'text',
        text: JSON.stringify({ state, message: 'missing calibration' }) }] }),
      _showToast: message => messages.push(message),
    });
    vm.runInContext(parseReply + startDirection, context);
    assert.equal(await context._startDirectionSensor('bumi', 'sound_direction', 'card-1'),
      state === 'running');
    assert.equal(messages.length, state === 'running' ? 0 : 1);
  });
}
