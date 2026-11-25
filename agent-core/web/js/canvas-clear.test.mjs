import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

// Exercise the actual canvas handlers with a small DOM and controllable network.
const source = readFileSync(new URL('./canvas.js', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
// By function name, as config-topic-refresh.test.mjs does — slicing between
// comment strings broke on unrelated comment edits.
function code(name) {
  const at = source.search(new RegExp(`(?:async )?function ${name}\\(`));
  assert.ok(at >= 0, `missing canvas function: ${name}`);
  // A one-line function ends on its own line; reading on to the next '\n}\n'
  // would pull in the function after it and shadow this file's stubs.
  const line = source.slice(at, source.indexOf('\n', at) + 1);
  if (line.split('{').length === line.split('}').length) return line;
  return source.slice(at, source.indexOf('\n}\n', at) + 3);
}
const handlers = ['_ensureEdit', '_projectEditLockReason', '_editLockReason', '_refuseEdit',
  '_editsLocked', '_clearCanvas', '_debouncedSave', '_saveLayout', '_writeLayout',
  '_syncEmptyState', '_removeCard', '_triggerAction'].map(code).join('\n');

function fixture({ confirmed = true, editor = true, running = false, known = true } = {}) {
  const removed = [], toasts = [], calls = [], stopped = [];
  const clearButton = { disabled: false };
  const cards = ['sensor', 'remote_mic', 'remote_camera'].map((toolName, i) => ({
    id: `card-${i}`, mcpId: 'device', toolName, driverName: 'Device', x: i, y: i,
    el: { remove: () => removed.push(toolName) },
  }));
  let confirms = 0, reloads = 0;
  const ctx = {
    _cards: cards,
    _allMcps: [{ id: 'device', tools: cards.map(c => ({ name: c.toolName, type: 'sensor' })) }],
    _connections: [{ fromCardId: 'card-0', toCardId: 'card-1' }],
    _execConnections: [{ fromCardId: 'card-1', toCardId: 'card-2' }],
    _isEditor: editor, _currentEditor: editor ? 'this-tab' : null,
    _sessionId: 'this-tab', _projectRunning: running, _projectStateKnown: known,
    _clearing: false, _layoutGen: 0, _saveTimer: null, _saveDeferred: false,
    _pendingLayoutSaves: new Set(), _zoom: 1.2, _tx: 10, _ty: 20, _emptyEl: { style: {} },
    document: { getElementById: id => id === 'canvas-clear' ? clearButton : null },
    confirm: () => { confirms++; return confirmed; },
    fetch: async (url, options) => {
      calls.push({ url, body: JSON.parse(options.body) });
      return { ok: true, status: 200, json: async () => ({}) };
    },
    _showToast: message => toasts.push(message),
    _logActivity() {}, _updateEditorUI() {}, _syncGeometryObservers() {}, _scheduleRedraw() {},
    _resolveAllTopics() {}, _autoStopOnDisconnect() {},
    _reloadLayout: async () => { reloads++; },
    _syncProjectState: async () => true,
    stopMicStream: () => stopped.push('mic'),
    stopCameraStream: () => stopped.push('camera'),
    setTimeout, clearTimeout, AbortController,
  };
  const api = runInNewContext(`${handlers}\n({ _clearCanvas, _removeCard, _saveLayout, _ensureEdit, _debouncedSave, _syncEmptyState })`, ctx);
  return { ctx, api, removed, toasts, calls, stopped, clearButton,
    confirms: () => confirms, reloads: () => reloads };
}

test('cancel leaves the shared layout untouched and does not claim the edit lock', async () => {
  const f = fixture({ confirmed: false, editor: false });
  await f.api._clearCanvas();
  assert.equal(f.confirms(), 1);
  assert.equal(f.calls.length, 0);
  assert.equal(f.ctx._cards.length, 3);
  assert.deepEqual(f.removed, []);
});

test('clear saves one empty layout, removes all cards and stops browser sources', async () => {
  const f = fixture();
  await f.api._clearCanvas();
  assert.equal(f.calls.length, 1);
  assert.equal(f.calls[0].url, '/api/canvas/layout');
  assert.deepEqual(f.calls[0].body, {
    cards: [], connections: [], execConnections: [],
    transform: { zoom: 1.2, tx: 10, ty: 20 }, session_id: 'this-tab',
  });
  assert.deepEqual(f.removed, ['sensor', 'remote_mic', 'remote_camera']);
  assert.deepEqual(f.stopped, ['mic', 'camera']);
  assert.equal(f.ctx._cards.length, 0);
  assert.equal(f.ctx._connections.length, 0);
  assert.equal(f.ctx._execConnections.length, 0);
  assert.equal(f.clearButton.disabled, true);
  assert.ok(f.toasts.includes('画布已清空'));
});

test('unknown or running project state refuses to clear before confirmation', async () => {
  for (const options of [{ running: true }, { known: false }]) {
    const f = fixture(options);
    await f.api._clearCanvas();
    assert.equal(f.confirms(), 0);
    assert.equal(f.calls.length, 0);
    assert.equal(f.ctx._cards.length, 3);
    assert.notEqual(f.toasts.length, 0);
  }
});

test('another editor prevents deletion', async () => {
  const f = fixture({ editor: false });
  f.ctx.fetch = async (url, options) => {
    f.calls.push({ url, body: JSON.parse(options.body) });
    return { ok: false, status: 423, json: async () => ({ editor: 'another-tab' }) };
  };
  await f.api._clearCanvas();
  assert.deepEqual(f.calls.map(c => c.url), ['/api/canvas/claim-edit']);
  assert.equal(f.ctx._cards.length, 3);
  assert.deepEqual(f.removed, []);
  // Read-only now: the button must not invite a click it can only refuse.
  assert.equal(f.clearButton.disabled, true);
});

test('a failed layout write keeps the cards visible and reports an uncertain result', async () => {
  const f = fixture();
  f.ctx.fetch = async () => { throw new Error('offline'); };
  await f.api._clearCanvas();
  assert.equal(f.ctx._cards.length, 3);
  assert.deepEqual(f.removed, []);
  assert.equal(f.reloads(), 1);
  assert.match(f.toasts.at(-1), /未能确认/);
  assert.equal(f.clearButton.disabled, false);
});

test('clear waits for a previous autosave before writing the empty layout', async () => {
  const f = fixture();
  let finishFirst;
  f.ctx.fetch = (url, options) => {
    f.calls.push({ url, body: JSON.parse(options.body) });
    if (f.calls.length === 1) return new Promise(resolve => { finishFirst = resolve; });
    return Promise.resolve({ ok: true, status: 200 });
  };
  const earlierSave = f.api._saveLayout();
  const clearing = f.api._clearCanvas();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.calls.length, 1);
  assert.equal(f.clearButton.disabled, true);
  assert.equal(await f.api._ensureEdit(), false);
  finishFirst({ ok: true, status: 200 });
  await Promise.all([earlierSave, clearing]);
  assert.equal(f.calls.length, 2);
  assert.equal(f.calls[0].body.cards.length, 3);
  assert.equal(f.calls[1].body.cards.length, 0);
});

test('a project that starts during clear preparation prevents the layout write', async () => {
  const f = fixture();
  f.ctx._syncProjectState = async () => { f.ctx._projectRunning = true; return true; };
  await f.api._clearCanvas();
  assert.equal(f.calls.length, 0);
  assert.equal(f.ctx._cards.length, 3);
  assert.match(f.toasts.at(-1), /停止智能控制/);
});

test('a write that times out but did land is reported as cleared and stops browser sources', async () => {
  const f = fixture();
  f.ctx.fetch = async () => { throw new Error('aborted'); };
  // The server saved [] before its slow teardown; reloading reads that back.
  f.ctx._reloadLayout = async () => {
    for (const c of f.ctx._cards) c.el.remove();
    f.ctx._cards = []; f.ctx._connections = []; f.ctx._execConnections = [];
    f.ctx._layoutGen++;
  };
  await f.api._clearCanvas();
  assert.deepEqual(f.stopped, ['mic', 'camera']);
  assert.equal(f.ctx._cards.length, 0);
  assert.equal(f.toasts.at(-1), '画布已清空');
});

test('a 403 reloads once and says the edit lock was lost, not the network', async () => {
  const f = fixture();
  f.ctx.fetch = async (url, options) => {
    f.calls.push({ url, body: JSON.parse(options.body) });
    return { ok: false, status: 403, json: async () => ({}) };
  };
  await f.api._clearCanvas();
  assert.equal(f.reloads(), 1);
  assert.match(f.toasts.at(-1), /编辑权/);
  assert.deepEqual(f.stopped, []);
});

test('a pending autosave is flushed, not dropped, when the clear is then aborted', async () => {
  const f = fixture();
  f.ctx._syncProjectState = async () => { f.ctx._projectRunning = true; return true; };
  f.api._debouncedSave();
  await f.api._clearCanvas();
  assert.equal(f.calls.length, 1);
  assert.equal(f.calls[0].body.cards.length, 3);
  assert.equal(f.ctx._cards.length, 3);
});

test('saves attempted during the clear are replayed once it ends', async () => {
  const f = fixture();
  f.ctx._syncProjectState = async () => {
    f.api._saveLayout();  // e.g. a pan while the clear waits
    f.ctx._projectRunning = true;
    return true;
  };
  await f.api._clearCanvas();
  assert.equal(f.calls.length, 0);
  assert.equal(f.ctx._saveDeferred, false);
  assert.notEqual(f.ctx._saveTimer, null);
  clearTimeout(f.ctx._saveTimer);
});

test('a reload during the clear wait aborts instead of wiping the new layout', async () => {
  const f = fixture();
  f.ctx._syncProjectState = async () => { f.ctx._layoutGen++; return true; };
  await f.api._clearCanvas();
  assert.equal(f.calls.length, 0);
  assert.equal(f.ctx._cards.length, 3);
  assert.match(f.toasts.at(-1), /更新/);
});

test('the clear button is disabled while read-only or running', () => {
  for (const state of [{ _isEditor: false, _currentEditor: 'other' }, { _projectRunning: true },
                       { _projectStateKnown: false }]) {
    const f = fixture();
    Object.assign(f.ctx, state);
    f.api._syncEmptyState();
    assert.equal(f.clearButton.disabled, true, JSON.stringify(state));
  }
  const f = fixture({ editor: false });  // nobody holds the lock: clicking claims it
  f.api._syncEmptyState();
  assert.equal(f.clearButton.disabled, false);
});

test('single-card removal skips stop only for an explicit exemption', async () => {
  for (const [tool, shouldStop] of [
    [{ name: 'sensor', type: 'sensor', inputSchema: { 'x-stop': { keep_running: true } } }, false],
    [{ name: 'sensor', type: 'sensor' }, true],
    [{ name: 'camera_head', type: 'sensor', multiInstance: false }, true],
    [{ name: 'sensor', type: 'sensor', multiInstance: true }, true],
    [{ name: 'sensor', type: 'processor', multiInstance: true }, true],
    [{ name: 'sensor', type: 'actuator' }, true],
    ['sensor', true],
    ...[false, 'true', 1, null].map(keep_running => [
      { name: 'sensor', type: 'sensor', inputSchema: { 'x-stop': { keep_running } } }, true]),
  ]) {
    const f = fixture();
    f.ctx._cards[0].toolName = typeof tool === 'string' ? tool : tool.name;
    f.ctx._allMcps[0].tools = [tool];
    await f.api._removeCard('card-0');
    const stops = f.calls.filter(c => c.url === '/api/mcp/device/call');
    assert.deepEqual(stops.map(c => c.body.arguments), shouldStop
      ? [{ action: 'stop', instance_id: 'card-0' }] : [], JSON.stringify(tool));
    assert.deepEqual(f.removed, ['sensor']);
    assert.equal(f.calls.at(-1).body.cards.length, 2);
  }
});

test('removing a browser source closes its local stream', async () => {
  for (const [id, source] of [['card-1', 'mic'], ['card-2', 'camera']]) {
    const f = fixture();
    await f.api._removeCard(id);
    assert.deepEqual(f.stopped, [source]);
  }
});
