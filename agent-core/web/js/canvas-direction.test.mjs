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
    _triggerAction: (...args) => calls.push(args),
  });
  vm.runInContext(addCard, context);
  context._addCard({ id: 'card-1', mcpId: 'bumi', toolName: tool.name }, save);
  return calls;
}

const direction = { name: 'sound_direction', type: 'sensor',
  topic_out: [{ format: 'sensor/sound-direction', topic: '/ubuntu/sound_direction/events' }] };

test('direction follows other boot-managed sensors without an extra start call', () => {
  assert.equal(drop(direction).length, 0);
});

test('restoring a saved layout does not undo an explicit project stop', () => {
  assert.equal(drop(direction, false).length, 0);
});

test('dropping unrelated cards does not start their workers', () => {
  assert.equal(drop({ name: 'asr', type: 'processor', multiInstance: true }).length, 0);
  assert.equal(drop({ name: 'mic', type: 'sensor', topic_out: [{ format: 'audio/pcm-16k' }] }).length, 0);
});
