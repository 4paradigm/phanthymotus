import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('./canvas.js', import.meta.url), 'utf8');
const names = ['_ensureEdit', '_claimEdit', '_requestEditor', '_setEditorState',
  '_releaseEdit', '_applyEditorState', '_checkEditStatus', '_reloadLayout', '_withDropEdit'];
const functions = names.map(name => source.match(
  new RegExp(`^(?:async )?function ${name}\\([^]*?^\\}`, 'm'),
)?.[0] || '').join('\n');

function fixture(fetch) {
  const messages = [];
  const context = vm.createContext({
    fetch, AbortController, setTimeout, clearTimeout,
    _showToast: message => messages.push(message),
    _updateEditorUI() {}, _scheduleReload() {}, _scheduleRedraw() {},
    _resolveAllTopics() {}, _syncEmptyState() {},
    _addCard() {},
  });
  vm.runInContext(`
    const _sessionId = 'session-a';
    const _EDIT_REQUEST_TIMEOUT_MS = 30;
    let _isEditor = false, _currentEditor = null, _editorRevision = 0;
    let _editClaimPromise = null;
    let _dropEditLease = null;
    let _cards = [], _connections = [], _execConnections = [];
    ${functions}
  `, context);
  return { context, messages, read: expression => vm.runInContext(expression, context) };
}

function response(status, data) {
  return { ok: status >= 200 && status < 300, status, json: async () => data };
}

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

test('an older layout response cannot overwrite a newly granted edit lock or cards', async () => {
  const layout = deferred();
  const env = fixture(url => url.endsWith('/layout') ? layout.promise
    : Promise.resolve(response(200, { editor: 'session-a' })));
  const reload = env.context._reloadLayout();
  assert.equal(await env.context._ensureEdit(), true);
  env.read("_cards.push({id: 'new-card', el: {remove() {}}})");
  layout.resolve(response(200, { editor: null, data: { cards: [] } }));
  await reload;
  assert.equal(env.read('_isEditor'), true);
  assert.equal(env.read('_cards.length'), 1);
});

test('an older status poll cannot undo a newer WebSocket grant', async () => {
  const status = deferred();
  const env = fixture(() => status.promise);
  const poll = env.context._checkEditStatus();
  env.context._applyEditorState('session-a', 'claim');
  status.resolve(response(200, { editor: null }));
  await poll;
  assert.equal(env.read('_isEditor'), true);
});

test('an older layout finishing before the claim reply cannot reject a grant without WebSocket', async () => {
  const layout = deferred();
  const claim = deferred();
  const env = fixture(url => url.endsWith('/layout') ? layout.promise : claim.promise);
  const reload = env.context._reloadLayout();
  const pending = env.context._ensureEdit();
  layout.resolve(response(200, { editor: null, data: { cards: [] } }));
  await reload;
  claim.resolve(response(200, { editor: 'session-a' }));
  assert.equal(await pending, true);
  assert.equal(env.read('_isEditor'), true);
});

test('concurrent edit attempts share one permission request', async () => {
  const claim = deferred();
  let requests = 0;
  const env = fixture(() => { requests++; return claim.promise; });
  const first = env.context._ensureEdit();
  const second = env.context._ensureEdit();
  assert.equal(requests, 1);
  claim.resolve(response(200, { editor: 'session-a' }));
  assert.deepEqual(await Promise.all([first, second]), [true, true]);
});

test('a failed release preserves the actual owner and permits retry', async () => {
  let releaseAttempts = 0;
  const env = fixture(url => {
    if (url.endsWith('/release-edit')) {
      releaseAttempts++;
      return Promise.resolve(response(releaseAttempts === 1 ? 500 : 200, {}));
    }
    return Promise.resolve(response(200, { editor: 'session-a' }));
  });
  env.context._applyEditorState('session-a', 'claim');
  await env.context._releaseEdit();
  assert.equal(env.read('_isEditor'), true);
  assert.ok(env.messages.some(message => message.includes('500')));
  await env.context._releaseEdit();
  assert.equal(env.read('_isEditor'), false);
});

test('server and authentication failures are reported with their own cause', async () => {
  for (const status of [401, 403, 500]) {
    const env = fixture(url => Promise.resolve(response(status,
      url.includes('edit-status') ? {} : { detail: 'request rejected' })));
    assert.equal(await env.context._ensureEdit(), false);
    assert.equal(env.read('_isEditor'), false);
    assert.ok(env.messages.some(message => message.includes(String(status))));
    assert.ok(!env.messages.some(message => message.includes('其他用户编辑')));
  }
});

test('a stalled permission request times out and the next attempt can succeed', { timeout: 1000 }, async () => {
  let requests = 0;
  const env = fixture((url, options) => {
    if (url.includes('edit-status')) return Promise.resolve(response(200, { editor: null }));
    requests++;
    if (requests > 1) return Promise.resolve(response(200, { editor: 'session-a' }));
    return new Promise((resolve, reject) => options.signal?.addEventListener('abort', () => {
      const error = new Error('Aborted'); error.name = 'AbortError'; reject(error);
    }));
  });
  assert.equal(await env.context._ensureEdit(), false);
  assert.ok(env.messages.some(message => message.includes('超时')));
  assert.equal(await env.context._ensureEdit(), true);
});

test('a successful HTTP response must confirm ownership for this session', async () => {
  const env = fixture(url => Promise.resolve(response(200,
    url.includes('edit-status') ? { editor: null } : { code: 200 })));
  assert.equal(await env.context._ensureEdit(), false);
  assert.equal(env.read('_isEditor'), false);
});

test('an older claim response cannot undo a newer release', async () => {
  const claim = deferred();
  const env = fixture(() => claim.promise);
  const pending = env.context._ensureEdit();
  env.context._applyEditorState('session-a', 'claim');
  env.context._applyEditorState(null, 'release');
  claim.resolve(response(200, { editor: 'session-a' }));
  assert.equal(await pending, false);
  assert.equal(env.read('_isEditor'), false);
});

function dropFixture() {
  const requests = [];
  const claim = deferred();
  const env = fixture(url => {
    requests.push(url);
    return url.endsWith('/claim-edit') ? claim.promise
      : Promise.resolve(response(200, {}));
  });
  return { ...env, requests, grant: () => claim.resolve(response(200, { editor: 'session-a' })) };
}

test('a rejected drop releases the edit lock it acquired', async () => {
  const env = dropFixture();
  const drop = env.context._withDropEdit(() => false);
  env.grant();
  await drop;
  assert.equal(env.requests.filter(url => url.endsWith('/release-edit')).length, 1);
  assert.equal(env.read('_isEditor'), false);
});

test('a rejected drop preserves a lock the session already held', async () => {
  const env = dropFixture();
  env.context._applyEditorState('session-a', 'claim');
  await env.context._withDropEdit(() => false);
  assert.equal(env.requests.length, 0);
  assert.equal(env.read('_isEditor'), true);
});

for (const firstAccepted of [false, true]) {
  test(`a rejected concurrent drop preserves a successful ${firstAccepted ? 'earlier' : 'later'} drop's lock`, async () => {
    const env = dropFixture();
    const first = env.context._withDropEdit(() => firstAccepted);
    const second = env.context._withDropEdit(() => !firstAccepted);
    env.grant();
    await Promise.all([first, second]);
    assert.equal(env.requests.filter(url => url.endsWith('/claim-edit')).length, 1);
    assert.equal(env.requests.filter(url => url.endsWith('/release-edit')).length, 0);
    assert.equal(env.read('_isEditor'), true);
  });
}

test('concurrent rejected drops release their shared temporary lock exactly once', async () => {
  const env = dropFixture();
  const drops = [env.context._withDropEdit(() => false), env.context._withDropEdit(() => false)];
  env.grant();
  await Promise.all(drops);
  assert.equal(env.requests.filter(url => url.endsWith('/claim-edit')).length, 1);
  assert.equal(env.requests.filter(url => url.endsWith('/release-edit')).length, 1);
  assert.equal(env.read('_isEditor'), false);
});

test('a rejected drop preserves a concurrent explicit edit request', async () => {
  const env = dropFixture();
  const drop = env.context._withDropEdit(() => false);
  const edit = env.context._ensureEdit();
  env.grant();
  await drop;
  assert.equal(await edit, true);
  assert.equal(env.requests.filter(url => url.endsWith('/release-edit')).length, 0);
  assert.equal(env.read('_isEditor'), true);
});

test('a rejected drop preserves an explicit edit request that was already pending', async () => {
  const env = dropFixture();
  const edit = env.context._ensureEdit();
  const drop = env.context._withDropEdit(() => false);
  env.grant();
  await drop;
  assert.equal(await edit, true);
  assert.equal(env.requests.filter(url => url.endsWith('/release-edit')).length, 0);
  assert.equal(env.read('_isEditor'), true);
});

for (const operation of ['edit', 'drop']) {
  test(`a new ${operation} waits for temporary-lock cleanup before claiming again`, { timeout: 1000 }, async () => {
    const releasing = deferred();
    const releaseStarted = deferred();
    const requests = [];
    const env = fixture(url => {
      requests.push(url);
      if (url.endsWith('/release-edit')) {
        releaseStarted.resolve();
        return releasing.promise;
      }
      return Promise.resolve(response(200, { editor: 'session-a' }));
    });
    const drop = env.context._withDropEdit(() => false);
    await releaseStarted.promise;
    const edit = operation === 'edit' ? env.context._ensureEdit()
      : env.context._withDropEdit(() => true);
    assert.equal(requests.filter(url => url.endsWith('/claim-edit')).length, 1);
    releasing.resolve(response(200, {}));
    await drop;
    assert.equal(await edit, true);
    assert.equal(requests.filter(url => url.endsWith('/claim-edit')).length, 2);
    assert.equal(env.read('_isEditor'), true);
  });
}
