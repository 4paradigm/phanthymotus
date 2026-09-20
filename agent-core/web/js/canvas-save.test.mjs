import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

// Exercise the actual browser save handler with an HTTP conflict response.
const source = readFileSync(new URL('./canvas.js', import.meta.url), 'utf8');
const start = source.indexOf('async function _saveLayout()');
const end = source.indexOf('// ── Helpers', start);
assert.ok(start >= 0 && end > start);

test('a rejected layout save displays the reason and restores the saved layout', async () => {
  const events = [];
  const context = vm.createContext({
    _isEditor: true, _cards: [], _connections: [], _execConnections: [],
    _zoom: 1, _tx: 0, _ty: 0, _sessionId: 'editor',
    fetch: async () => ({ status: 409, json: async () => ({ message: '请先停止智能控制后修改画布' }) }),
    _showToast: message => events.push(message),
    _reloadLayout: async () => events.push('reload'),
  });
  vm.runInContext(source.slice(start, end), context);
  await context._saveLayout();
  assert.deepEqual(events, ['请先停止智能控制后修改画布', 'reload']);
  assert.equal(context._isEditor, true);
});
