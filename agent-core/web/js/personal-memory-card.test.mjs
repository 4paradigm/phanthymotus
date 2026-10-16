import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('./sidebar.js', import.meta.url), 'utf8');
async function open(mcpId, toolName, data = {}) {
  const start = source.indexOf('export async function openToolConfigModal(');
  const bodyStart = source.indexOf('{', start) + 1;
  const end = source.indexOf('  // If all fields', bodyStart);
  const elements = {};
  const context = {
    isProjectRunning: () => false, ensureEdit: async () => true,
    _toolConfigs: { 'agentcore:decision_core': { personal_memory_enabled: false, llm_model: 'saved-model' } },
    document: {
      getElementById: id => elements[id] ||= { children: [], appendChild(child) { this.children.push(child); } },
      createElement: () => ({}),
    },
    fetch: async url => {
      assert.equal(url, '/api/canvas/tool-config/agentcore/decision_core');
      return { ok: true, json: async () => ({ data }) };
    },
    mcpId, toolName, configSchema: {},
  };
  return vm.runInNewContext(`(async () => {${source.slice(bodyStart, end)}
    return { savedValues, notes: bodyEl.children.map(n => n.textContent) }; })()`, context);
}

test('decision_core loads canonical memory switch and pending restart notice', async () => {
  const result = await open('agentcore', 'decision_core', {
    personal_memory_enabled: true, personal_memory_active: false, personal_memory_restart_required: true,
  });
  assert.equal(result.savedValues.personal_memory_enabled, true);
  assert.equal(result.savedValues.llm_model, 'saved-model');
  assert.match(result.notes[0], /自动读取画布连线/);
  assert.match(result.notes[0], /尚未生效/);
});

test('other card modals do not gain the memory notice or request', async () => {
  const result = await open('perception', 'asr');
  assert.equal(result.notes.length, 0);
});
