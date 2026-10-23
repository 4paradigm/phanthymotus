"""Card presentation metadata must not change the TTS API or manual routing.

The Node harness runs the real renderer, registered click handlers and request
builder with a small DOM stub. No browser package, ROS runtime or TTS model is
needed. Run with pytest; the UI cases require Node on PATH.
"""

import ast
import copy
import json
import pathlib
import shutil
import subprocess

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
CANVAS = ROOT / "agent-core/web/js/canvas.js"


@pytest.fixture
def tts_tool():
    # Evaluate only the declarative assignment, including its option-title
    # comprehension, without importing ROS or any synthesis engine.
    path = ROOT / "perception/plugins/tts.py"
    tree = ast.parse(path.read_text())
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "TOOLS"
                for target in node.targets)
    )
    namespace = {}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"),
         namespace)
    return next(tool for tool in namespace["TOOLS"] if tool["name"] == "tts")


def test_card_metadata_preserves_real_mcp_schema(tts_tool):
    import mcp_client

    schema = tts_tool["inputSchema"]
    assert schema["x-card-hidden-params"] == ["instance_id", "turn_id"]
    assert schema["x-card-hidden-actions"] == ["language"]
    assert {"instance_id", "turn_id"} <= schema["properties"].keys()
    assert "language" in schema["properties"]["action"]["enum"]

    original = copy.deepcopy(tts_tool)
    converted = mcp_client._to_openai_schema("perception", tts_tool)
    by_action = {item["name"].rsplit("__", 1)[1]: item["parameters"]
                 for item in converted}
    assert set(by_action["speak"]["properties"]) == {"text", "instance_id", "turn_id"}
    assert by_action["speak"]["required"] == ["text"]
    assert set(by_action["interrupt"]["properties"]) == {"instance_id"}
    assert "instance_id" in by_action["language"]["properties"]
    assert by_action["language"]["required"] == ["instance_id"]
    assert tts_tool == original

    without_card_metadata = copy.deepcopy(tts_tool)
    for key in ("x-card-hidden-params", "x-card-hidden-actions"):
        without_card_metadata["inputSchema"].pop(key)
    assert converted == mcp_client._to_openai_schema("perception", without_card_metadata)


NODE_HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {tool, scenario} = JSON.parse(fs.readFileSync(0, 'utf8'));

// Only the DOM operations used by card rendering, action changes and execution.
// Parse the renderer's HTML so an omitted field truly has no DOM input to read.
class Element {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.listeners = {};
    this.className = '';
    this.type = '';
    this._value = undefined;
  }
  appendChild(child) {
    child.parentElement = this;
    this.children.push(child);
    return child;
  }
  remove() {
    const siblings = this.parentElement.children;
    siblings.splice(siblings.indexOf(this), 1);
  }
  set innerHTML(html) {
    this.children = [];
    const stack = [this];
    for (const token of html.matchAll(/<\/?([\w-]+)\b([^>]*)>/g)) {
      const [raw, tag, attrs] = token;
      if (raw.startsWith('</')) { stack.pop(); continue; }
      const child = new Element(tag);
      for (const [, key, value] of attrs.matchAll(/([\w-]+)="([^"]*)"/g)) {
        if (key === 'class') child.className = value;
        else if (key.startsWith('data-')) {
          child.dataset[key.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = value;
        } else child[key] = value;
      }
      stack.at(-1).appendChild(child);
      if (!raw.endsWith('/>') && !['input', 'hr', 'br', 'img'].includes(tag)) stack.push(child);
    }
  }
  get value() {
    return this._value ?? (this.tagName === 'select'
      ? this.querySelector('option')?.value ?? '' : '');
  }
  set value(value) { this._value = value; }
  matches(selector) {
    const tag = selector.match(/^[\w-]+/)?.[0];
    if (tag && this.tagName !== tag) return false;
    for (const [, cls] of selector.matchAll(/\.([\w-]+)/g)) {
      if (!this.className.split(/\s+/).includes(cls)) return false;
    }
    for (const [, key, value] of selector.matchAll(/\[data-([\w-]+)="([^"]*)"\]/g)) {
      if (this.dataset[key] !== value) return false;
    }
    return true;
  }
  querySelectorAll(selector) {
    return this.children.flatMap(child => [
      ...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector),
    ]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
  closest(selector) {
    return this.matches(selector) ? this : this.parentElement?.closest(selector) ?? null;
  }
  addEventListener(type, callback) { (this.listeners[type] ??= []).push(callback); }
  async dispatch(type) {
    for (const callback of this.listeners[type] ?? []) {
      await callback({target: this, stopPropagation() {}, preventDefault() {}});
    }
  }
}

const requests = [];
const context = vm.createContext({
  document: {
    createElement: tag => new Element(tag), getElementById: () => null, addEventListener() {},
  },
  window: {addEventListener() {}},
  sessionId: () => 'test-editor',
  setInterval() {},
  requestAnimationFrame: () => 1,
  fetch: async (url, options) => {
    requests.push({url, ...JSON.parse(options.body)});
    return {ok: true, json: async () => ({code: 200, data: 'ok'})};
  },
  tool,
});
// Keep every function body intact; imports belong to unrelated UI subsystems.
const source = fs.readFileSync(process.argv[1], 'utf8')
  .replace(/^import\s[\s\S]*?\sfrom\s+['"][^'"]+['"];?/gm, '')
  .replace(/^export\s+/gm, '');
vm.runInContext(source + '\n_isEditor = true; _projectRunning = true;', context);

function render(id = 'tts-card-a', definition = tool) {
  context.definition = definition;
  context.cardId = id;
  return vm.runInContext(`
    _allMcps = [{id: 'perception', tools: [definition]}];
    _buildCardEl({id: cardId, mcpId: 'perception', toolName: definition.name,
                  driverName: 'perception', x: 0, y: 0});
  `, context);
}
const input = (card, key) => card.querySelector('.canvas-field-input[data-key="' + key + '"]');
const actions = card => input(card, 'action').querySelectorAll('option').map(o => o.value);
async function click(card) {
  const before = requests.length;
  await card.querySelector('.canvas-exec-btn').dispatch('click');
  assert.equal(requests.length, before + 1, 'the real click handler must send one request');
  assert.equal(requests.at(-1).url, '/api/mcp/perception/call');
  return requests.at(-1).arguments;
}
function inject(card, key, value) {
  const field = new Element('input');
  field.className = 'canvas-field-input';
  field.dataset.key = key;
  field.value = value;
  card.appendChild(field);
}

(async () => {
  if (scenario === 'tts') {
    const card = render();
    assert.deepEqual(actions(card), ['speak', 'interrupt']);
    assert.equal(card.querySelector('[data-key="instance_id"]'), null);
    assert.equal(card.querySelector('[data-key="turn_id"]'), null);
    const visible = card.querySelectorAll('.canvas-field')
      .filter(field => field.style.display !== 'none')
      .map(field => field.querySelector('.canvas-field-input').dataset.key);
    assert.deepEqual(visible, ['action', 'text']);
    input(card, 'text').value = 'Hello，世界';
    assert.deepEqual(await click(card), {
      action: 'speak', text: 'Hello，世界', instance_id: 'tts-card-a',
    });

    // A stale or injected field must not override the card's routing identity.
    inject(card, 'instance_id', 'wrong-card');
    inject(card, 'turn_id', 'stale-language-turn');
    assert.deepEqual(await click(card), {
      action: 'speak', text: 'Hello，世界', instance_id: 'tts-card-a',
    });

    input(card, 'text').value = '';
    input(card, 'action').value = 'interrupt';
    await input(card, 'action').dispatch('change');
    assert.equal(input(card, 'text').closest('.canvas-field').style.display, 'none');
    assert.deepEqual(await click(card), {action: 'interrupt', instance_id: 'tts-card-a'});

    const second = render('tts-card-b');
    input(second, 'text').value = 'Second card';
    assert.deepEqual(await click(second), {
      action: 'speak', text: 'Second card', instance_id: 'tts-card-b',
    });
    assert.equal(requests.at(-1).tool, 'tts');
  } else if (scenario === 'no-metadata') {
    delete tool.inputSchema['x-card-hidden-params'];
    delete tool.inputSchema['x-card-hidden-actions'];
    tool.name = 'another-processor';
    const card = render('unrelated-card');
    assert.deepEqual(actions(card), ['speak', 'interrupt', 'language']);
    assert.ok(input(card, 'instance_id'));
    assert.ok(input(card, 'turn_id'));
    input(card, 'text').value = 'Legacy behavior';
    assert.deepEqual(await click(card), {action: 'speak', text: 'Legacy behavior'});
    input(card, 'instance_id').value = 'manual-instance';
    input(card, 'turn_id').value = 'manual-turn';
    assert.deepEqual(await click(card), {
      action: 'speak', text: 'Legacy behavior', instance_id: 'manual-instance', turn_id: 'manual-turn',
    });
  } else if (scenario === 'opt-in') {
    // Metadata is generic; hiding another field alone must not opt into routing.
    tool.name = 'custom-processor';
    tool.inputSchema['x-card-hidden-params'] = ['turn_id'];
    const card = render('custom-card');
    assert.equal(input(card, 'turn_id'), null);
    assert.ok(input(card, 'instance_id'));
    assert.deepEqual(actions(card), ['speak', 'interrupt']);
    input(card, 'text').value = 'Custom';
    inject(card, 'turn_id', 'unrelated-turn');
    assert.deepEqual(await click(card), {action: 'speak', text: 'Custom'});
  } else throw new Error('Unknown scenario: ' + scenario);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


@pytest.mark.parametrize("scenario", ["tts", "no-metadata", "opt-in"])
def test_real_canvas_render_and_click(tts_tool, scenario):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to run the real canvas renderer")
    result = subprocess.run(
        [node, "-e", NODE_HARNESS, str(CANVAS)],
        input=json.dumps({"tool": tts_tool, "scenario": scenario}),
        text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
