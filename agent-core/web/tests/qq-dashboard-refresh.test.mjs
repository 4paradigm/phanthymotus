/** QQ setup must not appear automatically on Dashboard reload. */
import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';

const source = readFileSync(
  new URL('../js/channel-setup-entry.js', import.meta.url), 'utf8'
).replace(/^import\s+.*;\s*$/gm, '');

function bootDashboard(search) {
  const listeners = new Map();
  const opened = [];
  const timeouts = [];
  const makeElement = () => ({
    hidden: false,
    children: [],
    addEventListener(type, cb) {
      const handlers = listeners.get(type) || [];
      handlers.push(cb);
      listeners.set(type, handlers);
    },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    after() {},
    appendChild(node) { this.children.push(node); },
  });
  const elements = new Map([
    'channel-add-btn', 'channel-list', 'channel-users-panel',
    'channel-users-list', 'channel-close', 'btn-channels',
  ].map(id => [id, makeElement()]));
  const document = {
    readyState: 'complete',
    getElementById(id) { return elements.get(id) || null; },
    createElement() { return makeElement(); },
    addEventListener(type, cb) {
      const handlers = listeners.get(type) || [];
      handlers.push(cb);
      listeners.set(type, handlers);
    },
  };
  class MutationObserver { observe() {} }
  vm.runInNewContext(source, {
    document, window: {location: {search}},
    MutationObserver,
    initQQManagement() {},
    initQQSettingsUI() {},
    interceptLegacyQQConfigure() {},
    openQQWizard(options) { opened.push(options); },
    setTimeout(fn) { timeouts.push(fn); },
  }, {timeout: 2000});
  return {listeners, opened, timeouts};
}

for (const search of ['', '?qq-setup=1']) {
  test(`Dashboard refresh does not auto-open QQ setup for URL ${search || '(plain)'}`, () => {
    const state = bootDashboard(search);
    assert.equal(state.opened.length, 0);
    assert.equal(state.timeouts.length, 0);
  });
}

test('Only manually choosing QQ opens the wizard, not other channels', () => {
  const state = bootDashboard('?qq-setup=1');
  const onChange = state.listeners.get('change')[0];
  onChange({target: {id: 'channel-form-platform', value: 'feishu'}});
  onChange({target: {id: 'channel-form-platform', value: 'telegram'}});
  assert.equal(state.opened.length, 0);
  onChange({target: {id: 'channel-form-platform', value: 'qq'}});
  assert.equal(state.opened.length, 1);
});
