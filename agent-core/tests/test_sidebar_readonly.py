"""Run the sidebar and drop handlers in Node without browser dependencies.

Regression: opening tool details must not claim the canvas. A sidebar drag
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
  constructor() {
    this.listeners = {};
    this.children = {};
    this.dataset = {};
    this.classes = new Set(['hidden']);
    this.classList = {
      add: name => this.classes.add(name),
      remove: name => this.classes.delete(name),
    };
  }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  querySelector(selector) {
    if (!this.children[selector]) {
      this.children[selector] = new Element();
      this.children[selector].parent = this;
    }
    return this.children[selector];
  }
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
  createElement: () => new Element(),
  getElementById: id => elements[id] ||= new Element(),
};
let claims = 0;
let canEdit = true;
const sandbox = {
  assert, document, console,
  ensureEdit: async () => { claims++; return canEdit; },
  isEditor: () => false,
  isProjectRunning: () => false,
  cardsForTool: () => [],
  isDisplayOnly: () => false,
};
vm.createContext(sandbox);
const run = code => vm.runInContext(code, sandbox);
const sidebar = fs.readFileSync(process.argv[1] + '/sidebar.js', 'utf8');
run(sidebar.replace(/^import[\s\S]*?;/gm, '').replace(/^export\s+/gm, ''));
run("_backdrop = document.getElementById('backdrop');");
const tool = { name: 'camera', type: 'sensor', description: 'Camera details',
  inputSchema: { type: 'object' } };
sandbox.tool = tool;
sandbox.mcp = { id: 'robot', server_name: 'Robot' };

// Execute the real drop handler, with rendering and lock I/O recorded separately.
const canvas = fs.readFileSync(process.argv[1] + '/canvas.js', 'utf8');
const dropSource = canvas.match(/^function _setupDropZone\(\) \{[\s\S]*?^\}/m)[0];
let running = false;
let added = [];
let rejected = [];
let finishClaim;
let configured = true;
const area = new Element();
Object.assign(sandbox, {
  _canvasEl: area, _cards: [], _tx: 0, _ty: 0, _zoom: 1,
  _editsLocked: () => running,
  _editLockReason: () => 'Project running',
  _showDropReject: (event, reason) => rejected.push(reason),
  _ensureEdit: () => {
    claims++;
    return new Promise(resolve => { finishClaim = resolve; });
  },
  isToolConfigured: () => configured,
  _findNonOverlappingPos: (x, y) => ({ x, y }),
  _addCard: data => added.push(data),
});
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
    """)


def test_drop_is_rejected_when_another_editor_holds_permission():
    run_js("""
      const pending = drop();
      finishClaim(false);
      await pending;
      assert.equal(claims, 1);
      assert.equal(added.length, 0);
    """)


def test_running_project_rejects_drop_before_claiming():
    run_js("""
      running = true;
      await drop();
      assert.equal(claims, 0);
      assert.equal(added.length, 0);
      assert.equal(rejected.length, 1);
    """)


def test_project_started_while_waiting_for_permission_rejects_drop():
    run_js("""
      const pending = drop();
      running = true;
      finishClaim(true);
      await pending;
      assert.equal(added.length, 0);
      assert.equal(rejected.length, 1);
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
    """)
