"""Run the sidebar and drop handlers in Node without browser dependencies.

Regression: viewing tool details/configuration must not claim the canvas. A sidebar drag
claims permission only when dropped, including when the claim is asynchronous.
Run with pytest and Node.js on PATH.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


WEB = Path(__file__).resolve().parents[1] / 'web' / 'js'
NODE = shutil.which('node')
pytestmark = pytest.mark.skipif(NODE is None, reason='Node.js is required')

HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tagName = 'div') {
    this.tagName = tagName.toUpperCase();
    this.listeners = {};
    this.children = {};
    this.childNodes = [];
    this.dataset = {};
    this.style = {};
    this.attributes = {};
    this.value = '';
    this.classes = new Set(['hidden']);
    this.classList = {
      add: name => this.classes.add(name),
      remove: name => this.classes.delete(name),
      contains: name => this.classes.has(name),
      toggle: (name, on = !this.classes.has(name)) => {
        if (on) this.classes.add(name); else this.classes.delete(name);
      },
    };
  }
  set className(value) { this.classes = new Set(value.split(/\s+/)); }
  get className() { return [...this.classes].join(' '); }
  set value(value) { this._value = String(value); }
  get value() { return this._value; }
  set innerHTML(value) { this.html = value; this.childNodes = []; this.children = {}; }
  get innerHTML() { return this.html || ''; }
  setAttribute(key, value) { this.attributes[key] = value; }
  appendChild(child) {
    child.parent = child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }
  append(...children) { children.forEach(child => this.appendChild(child)); }
  replaceChild(child, old) {
    this.childNodes[this.childNodes.indexOf(old)] = child;
    child.parent = child.parentNode = this;
    if (old.id) elements[old.id] = child;
  }
  cloneNode() {
    const clone = new Element(this.tagName);
    Object.assign(clone, { id: this.id, className: this.className,
      disabled: this.disabled, textContent: this.textContent });
    return clone;
  }
  matches(selector) {
    return selector.split(',').some(part => {
      const tag = part.match(/^[a-z]+/i)?.[0];
      if (tag && this.tagName !== tag.toUpperCase()) return false;
      for (const match of part.matchAll(/\.([\w-]+)/g)) {
        if (!this.classes.has(match[1])) return false;
      }
      for (const match of part.matchAll(/\[data-([\w-]+)(?:="([^"]*)")?\]/g)) {
        const key = match[1].replace(/-([a-z])/g, (_, c) => c.toUpperCase());
        if (!(key in this.dataset)) return false;
        if (match[2] !== undefined && this.dataset[key] !== match[2]) return false;
      }
      return true;
    });
  }
  querySelectorAll(selector) {
    return this.childNodes.flatMap(child => [
      ...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector),
    ]);
  }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  querySelector(selector) {
    const actual = this.querySelectorAll(selector)[0];
    if (actual) return actual;
    if (!this.children[selector]) {
      this.children[selector] = new Element();
      this.children[selector].parent = this;
    }
    return this.children[selector];
  }
  closest(selector) { return this.matches(selector) ? this : this.parent?.closest(selector); }
  async fire(name, event = {}) {
    event.preventDefault ||= () => { event.defaultPrevented = true; };
    event.stopPropagation ||= () => { event.stopped = true; };
    for (const callback of this.listeners[name] || []) await callback(event);
    if (this.parent && !event.stopped) await this.parent.fire(name, event);
    return event;
  }
  getBoundingClientRect() { return { left: 0, top: 0 }; }
}

const elements = {};
const document = {
  createElement: tag => new Element(tag),
  getElementById: id => {
    if (!elements[id]) { elements[id] = new Element(); elements[id].id = id; }
    return elements[id];
  },
};
let claims = 0;
let canEdit = true;
let editing = false;
let projectActive = false;
let mediaRequests = 0;
const requests = [];
const sandbox = {
  assert, document, console,
  ensureEdit: async () => { claims++; return canEdit; },
  isEditor: () => editing,
  isProjectRunning: () => projectActive,
  cardsForTool: () => [],
  isDisplayOnly: () => false,
  fetch: async (url, options = {}) => {
    requests.push({ url, ...options });
    return { ok: true, json: async () => ({ data: {} }) };
  },
  alert: message => { throw new Error(message); },
  navigator: { mediaDevices: {
    getUserMedia: async () => { mediaRequests++; return { getTracks: () => [] }; },
    enumerateDevices: async () => [],
  } },
};
vm.createContext(sandbox);
const run = code => vm.runInContext(code, sandbox);
const sidebar = fs.readFileSync(process.argv[1] + '/sidebar.js', 'utf8');
run(sidebar.replace(/^import[\s\S]*?;/gm, '').replace(/^export\s+/gm, ''));
run("_backdrop = document.getElementById('backdrop'); _scroll = document.getElementById('sidebar-scroll');");
const configOverlay = document.getElementById('tool-config-overlay');
configOverlay.append(document.getElementById('tool-config-title'),
  document.getElementById('tool-config-body'), document.getElementById('tool-config-save'),
  document.getElementById('tool-config-close'), document.getElementById('tool-config-cancel'));
const tool = { name: 'camera', type: 'sensor', description: 'Camera details',
  inputSchema: { type: 'object' } };
sandbox.tool = tool;
sandbox.mcp = { id: 'robot', server_name: 'Robot' };

// Execute the real drop handler, with rendering and lock I/O recorded separately.
const canvas = fs.readFileSync(process.argv[1] + '/canvas.js', 'utf8');
const dropSource = canvas.match(/^function _setupDropZone\(\) \{[\s\S]*?^\}/m)[0];
const dropEditSource = canvas.match(/^async function _withDropEdit\(edit\) \{[\s\S]*?^\}/m)[0];
let running = false;
let added = [];
let rejected = [];
let releases = 0;
let finishClaim;
let configured = true;
const area = new Element();
Object.assign(sandbox, {
  _canvasEl: area, _cards: [], _tx: 0, _ty: 0, _zoom: 1,
  _isEditor: false, _editClaimPromise: null, _dropEditLease: null,
  _editsLocked: () => running,
  _editLockReason: () => 'Project running',
  _showDropReject: (event, reason) => rejected.push(reason),
  _ensureEdit: () => {
    if (sandbox._isEditor) return Promise.resolve(true);
    claims++;
    return new Promise(resolve => {
      finishClaim = granted => { sandbox._isEditor = granted; resolve(granted); };
    });
  },
  _releaseEdit: async () => { releases++; sandbox._isEditor = false; },
  isToolConfigured: () => configured,
  _findNonOverlappingPos: (x, y) => ({ x, y }),
  _addCard: data => added.push(data),
});
run(dropEditSource);
run(dropSource);
run('_setupDropZone();');

function drop(data = { mcpId: 'robot', toolName: 'camera' }) {
  let readable = true;
  const promise = area.fire('drop', {
    clientX: 300, clientY: 200,
    dataTransfer: { getData: () => {
      assert.ok(readable, 'drag data was accessed after the drop event');
      return JSON.stringify(data);
    } },
  });
  readable = false;
  return promise;
}

(async () => {
SCENARIO
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


def run_js(scenario):
    result = subprocess.run(
        [NODE, '-e', HARNESS.replace('SCENARIO', scenario), str(WEB)],
        capture_output=True, text=True, encoding='utf-8', timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('builder', ['_buildChip', '_buildToolCard'])
@pytest.mark.parametrize('occupied', [False, True])
def test_details_do_not_claim_edit_permission(builder, occupied):
    run_js(f"""
      canEdit = {str(not occupied).lower()};
      const card = run('{builder}(mcp, tool)');
      const target = '{builder}' === '_buildChip'
        ? card : card.querySelector('.tool-card-info-btn');
      await target.fire('pointerdown');
      await target.fire('click');
      assert.equal(document.getElementById('tool-detail-title').textContent, 'camera');
      assert.ok(document.getElementById('tool-detail-body').innerHTML.includes('Camera details'));
      assert.equal(document.getElementById('backdrop').classes.has('hidden'), false);
      assert.equal(claims, 0);
    """)


@pytest.mark.parametrize('builder', ['_buildChip', '_buildToolCard'])
def test_drag_and_cancel_does_not_claim_permission(builder):
    run_js(f"""
      const card = run('{builder}(mcp, tool)');
      await card.fire('pointerdown');
      let payload;
      const event = await card.fire('dragstart', {{ dataTransfer: {{
        setData: (mime, data) => {{ payload = JSON.parse(data); }},
      }} }});
      assert.ok(!event.defaultPrevented, 'first drag was cancelled');
      assert.equal(payload.toolName, 'camera');
      await card.fire('dragend');
      assert.equal(card.classes.has('dragging-source'), false);
      assert.equal(claims, 0);
    """)


def test_drop_waits_for_permission_and_keeps_drag_payload():
    run_js("""
      const pending = drop();
      assert.equal(claims, 1);
      assert.equal(added.length, 0);
      finishClaim(true);
      await pending;
      assert.equal(added.length, 1);
      assert.equal(added[0].toolName, 'camera');
      assert.equal(added[0].x, 190);
      assert.equal(added[0].y, 176);
      assert.equal(releases, 0);
      assert.equal(sandbox._isEditor, true);
    """)


def test_drop_is_rejected_when_another_editor_holds_permission():
    run_js("""
      const pending = drop();
      finishClaim(false);
      await pending;
      assert.equal(claims, 1);
      assert.equal(added.length, 0);
      assert.equal(releases, 0);
    """)


def test_running_project_rejects_drop_before_claiming():
    run_js("""
      running = true;
      await drop();
      assert.equal(claims, 0);
      assert.equal(added.length, 0);
      assert.equal(rejected.length, 1);
      assert.equal(releases, 0);
    """)


def test_project_started_while_waiting_for_permission_rejects_drop():
    run_js("""
      const pending = drop();
      running = true;
      finishClaim(true);
      await pending;
      assert.equal(added.length, 0);
      assert.equal(rejected.length, 1);
      assert.equal(releases, 1);
      assert.equal(sandbox._isEditor, false);
    """)


@pytest.mark.parametrize('reason', ['duplicate', 'unconfigured'])
def test_drop_rechecks_tool_state_after_permission(reason):
    run_js(f"""
      const pending = drop({{ mcpId: 'robot', toolName: 'camera', hasConfig: true }});
      if ('{reason}' === 'duplicate') sandbox._cards.push({{ mcpId: 'robot', toolName: 'camera' }});
      else configured = false;
      finishClaim(true);
      await pending;
      assert.equal(added.length, 0);
      assert.equal(rejected.length, 1);
      assert.equal(releases, 1);
      assert.equal(sandbox._isEditor, false);
    """)


@pytest.mark.parametrize('reason', ['running', 'duplicate', 'unconfigured'])
def test_rejected_drop_preserves_preexisting_edit_permission(reason):
    run_js(f"""
      sandbox._isEditor = true;
      const pending = drop({{ mcpId: 'robot', toolName: 'camera', hasConfig: true }});
      if ('{reason}' === 'running') running = true;
      else if ('{reason}' === 'duplicate') sandbox._cards.push({{ mcpId: 'robot', toolName: 'camera' }});
      else configured = false;
      await pending;
      assert.equal(added.length, 0);
      assert.equal(rejected.length, 1);
      assert.equal(claims, 0);
      assert.equal(releases, 0);
      assert.equal(sandbox._isEditor, true);
    """)


@pytest.mark.parametrize('builder', ['_buildChip', '_buildToolCard'])
@pytest.mark.parametrize('occupied', [False, True])
def test_sidebar_gear_is_readonly_without_claiming_permission(builder, occupied):
    run_js(f"""
      canEdit = {str(not occupied).lower()};
      tool.configSchema = {{ properties: {{ mode: {{ type: 'string', default: 'default' }} }} }};
      run("_toolConfigs['robot:camera'] = {{ mode: 'saved' }}");
      const card = run('{builder}(mcp, tool)');
      const gear = card.querySelector('{'.chip-config-btn' if builder == '_buildChip' else '.tool-card-config-btn'}');
      await gear.fire('pointerdown');
      await gear.fire('click');
      assert.equal(configOverlay.classes.has('hidden'), false);
      assert.equal(run('_configAccess.fields.disabled'), true);
      assert.equal(run(`_configAccess.fields.querySelector('[data-key="mode"]').value`), 'saved');
      const save = document.getElementById('tool-config-save');
      assert.equal(save.disabled, true);
      assert.equal(save.classes.has('hidden'), true);
      // Invoke the handler directly too: the guard must hold beyond disabled UI.
      await save.fire('click');
      assert.equal(claims, 0);
      assert.equal(requests.length, 0);
      await document.getElementById('tool-config-cancel').fire('click');
      assert.equal(configOverlay.classes.has('hidden'), true);
      assert.equal(run("_toolConfigs['robot:camera'].mode"), 'saved');
    """)


@pytest.mark.parametrize('instance', [False, True])
def test_instance_and_shared_config_save_still_work_for_existing_editor(instance):
    run_js(f"""
      editing = true;
      const schema = {{ properties: {{
        mode: {{ type: 'string', default: 'shared' }},
        gain: {{ type: 'number', scope: 'instance', default: 0 }},
        tags: {{ type: 'array', items: {{ type: 'string' }}, scope: 'instance', default: ['a', 'b'] }},
      }} }};
      if ({str(instance).lower()}) await sandbox.openInstanceConfigModal('robot', 'camera', 'card-a', schema);
      else await sandbox.openToolConfigModal('robot', 'camera', schema);
      assert.equal(run('_configAccess.fields.disabled'), false);
      const save = document.getElementById('tool-config-save');
      assert.equal(save.disabled, false);
      assert.equal(save.classes.has('hidden'), false);
      run(`_configAccess.fields.querySelector('[data-key="mode"]').value = "changed"`);
      await save.fire('click');
      assert.equal(claims, 0);
      assert.equal(configOverlay.classes.has('hidden'), true);
      const writes = requests.filter(r => r.method === 'PUT');
      if ({str(instance).lower()}) {{
        assert.deepEqual(writes.map(r => [r.url, JSON.parse(r.body)]), [
          ['/api/canvas/tool-config/robot/camera/card-a', {{ gain: 0, tags: ['a', 'b'] }}],
          ['/api/canvas/tool-config/robot/camera', {{ mode: 'changed' }}],
        ]);
      }} else {{
        assert.deepEqual(writes.map(r => JSON.parse(r.body)), [{{ mode: 'changed' }}]);
      }}
    """)


@pytest.mark.parametrize('instance', [False, True])
def test_readonly_configuration_does_not_request_media_permission(instance):
    run_js(f"""
      const schema = {{ properties: {{
        mic: {{ type: 'string', format: 'audio-input-device', scope: 'instance' }},
        cam: {{ type: 'string', format: 'video-input-device', scope: 'instance' }},
      }} }};
      if ({str(instance).lower()}) await sandbox.openInstanceConfigModal('robot', 'camera', 'card-a', schema);
      else await sandbox.openToolConfigModal('robot', 'camera', schema);
      await document.getElementById('tool-config-save').fire('click');
      assert.equal(run('_configAccess.fields.disabled'), true);
      assert.equal(mediaRequests, 0);
      assert.equal(claims, 0);
      assert.equal(requests.length, 0);
    """)


@pytest.mark.parametrize('instance', [False, True])
@pytest.mark.parametrize('editor', [False, True])
def test_running_project_allows_readonly_configuration_view(instance, editor):
    run_js(f"""
      editing = {str(editor).lower()};
      projectActive = true;
      const schema = {{ properties: {{ mode: {{ type: 'string', default: 'saved' }} }} }};
      if ({str(instance).lower()}) await sandbox.openInstanceConfigModal('robot', 'camera', 'card-a', schema);
      else await sandbox.openToolConfigModal('robot', 'camera', schema);
      assert.equal(configOverlay.classes.has('hidden'), false);
      assert.equal(run('_configAccess.fields.disabled'), true);
      await document.getElementById('tool-config-save').fire('click');
      assert.equal(claims, 0);
      assert.equal(requests.length, 0);
    """)


@pytest.mark.parametrize('reason', ['lock_lost', 'project_started'])
@pytest.mark.parametrize('instance', [False, True])
def test_open_config_becomes_readonly_and_does_not_reclaim_lock(instance, reason):
    run_js(f"""
      editing = true;
      const schema = {{ properties: {{ mode: {{ type: 'string', default: 'saved' }} }} }};
      if ({str(instance).lower()}) await sandbox.openInstanceConfigModal('robot', 'camera', 'card-a', schema);
      else await sandbox.openToolConfigModal('robot', 'camera', schema);
      if ('{reason}' === 'lock_lost') editing = false;
      else projectActive = true;
      sandbox.syncToolConfigPermissions();
      assert.equal(run('_configAccess.fields.disabled'), true);
      await document.getElementById('tool-config-save').fire('click');
      assert.equal(claims, 0);
      assert.equal(requests.length, 0);
    """)


def test_readonly_config_does_not_silently_become_writable():
    run_js("""
      const schema = { properties: { mode: { type: 'string', default: 'saved' } } };
      await sandbox.openToolConfigModal('robot', 'camera', schema);
      editing = true;
      sandbox.syncToolConfigPermissions();
      assert.equal(run('_configAccess.fields.disabled'), true);
      await document.getElementById('tool-config-save').fire('click');
      assert.equal(requests.length, 0);
      await document.getElementById('tool-config-close').fire('click');
      await sandbox.openToolConfigModal('robot', 'camera', schema);
      assert.equal(run('_configAccess.fields.disabled'), false);
    """)


@pytest.mark.parametrize('editor', [False, True])
def test_decision_core_live_configuration_still_requires_existing_editor(editor):
    run_js(f"""
      editing = {str(editor).lower()};
      projectActive = true;
      await sandbox.openToolConfigModal('agentcore', 'decision_core', {{
        properties: {{ mode: {{ type: 'string', default: 'saved' }} }},
      }});
      assert.equal(run('_configAccess.fields.disabled'), {str(not editor).lower()});
      await document.getElementById('tool-config-save').fire('click');
      assert.equal(requests.length, {1 if editor else 0});
      assert.equal(claims, 0);
    """)


def test_save_rechecks_ownership_and_stops_remaining_scope_writes():
    run_js("""
      editing = true;
      await sandbox.openInstanceConfigModal('robot', 'camera', 'card-a', {
        properties: {
          mode: { type: 'string', default: 'shared' },
          gain: { type: 'number', scope: 'instance', default: 0 },
        },
      });
      sandbox.fetch = async (url, options) => {
        requests.push({ url, ...options });
        editing = false;
        return { ok: true };
      };
      await document.getElementById('tool-config-save').fire('click');
      assert.deepEqual(requests.map(r => r.url), ['/api/canvas/tool-config/robot/camera/card-a']);
      assert.equal(run('_configAccess.fields.disabled'), true);
      assert.equal(claims, 0);
    """)
