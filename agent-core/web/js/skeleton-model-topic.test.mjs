// The skeleton panel is per topic, but the driver's `model` tool used to be
// called with empty arguments. A driver that serves several cards (the left and
// right hand of one L20, for example) could then only ever give every panel the
// same URDF. The monitor dashboard now hands the panel's topic to the renderer,
// and the renderer forwards it with the model call. These tests pin both halves:
// the call sites and the request body.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const skeletonSource = readFileSync(new URL('./renderers/skeleton.js', import.meta.url), 'utf8');
const dashboardSource = readFileSync(new URL('./monitor-dashboard.js', import.meta.url), 'utf8');

// Slice one method out of the renderer source so it can run in a sandbox with
// stubs. Brace counting is enough here: the method has no braces in strings and
// only a balanced `${…}` inside its request URL template.
function methodSource(name) {
  const at = skeletonSource.search(new RegExp(`\\n  (?:async )?${name}\\(`));
  assert.ok(at >= 0, `${name} not found in renderers/skeleton.js`);
  const start = skeletonSource.indexOf('{', at);
  let depth = 0;
  for (let i = start; i < skeletonSource.length; i += 1) {
    if (skeletonSource[i] === '{') depth += 1;
    else if (skeletonSource[i] === '}') {
      depth -= 1;
      if (depth === 0) return skeletonSource.slice(at + 1, i + 1);
    }
  }
  assert.fail(`unterminated ${name} in renderers/skeleton.js`);
}

function loadModelSandbox() {
  const calls = [];
  const context = {
    console,
    encodeURIComponent,
    fetch: async (url, options) => {
      calls.push({ url, body: JSON.parse(options.body) });
      return { json: async () => ({ data: [{ text: JSON.stringify({ urdf: '<robot name="stub"/>' }) }] }) };
    },
  };
  const method = runInNewContext(`({ ${methodSource('_loadModel')} })`, context)._loadModel;
  const renderer = {
    _parseAndBuild(urdf) { this.urdf = urdf; },
    _buildFallbackSkeleton() { this.fallback = true; },
    _buildQuadrupedSkeleton() { this.quadruped = true; },
  };
  return { method, renderer, calls };
}

test('the monitor dashboard hands the panel topic to the renderer', () => {
  const mounts = [...dashboardSource.matchAll(
    /renderer\.mount\(body, _topicMcpMap\[topicPath\] \|\| 'dashboard', topicPath\)/g)];
  assert.equal(mounts.length, 2,
    'the first mount and the re-mount after a mode switch must both pass the topic');
});

test('the skeleton renderer passes the topic on to the model loader', () => {
  assert.match(skeletonSource, /mount\(container, mcpId, topicPath\)/);
  assert.match(skeletonSource, /this\._loadModel\(mcpId, topicPath\)/);
});

test('the model request carries the panel topic', async () => {
  const { method, renderer, calls } = loadModelSandbox();
  await method.call(renderer, 'linkerhand-l20', '/nvidia_desktop/hand/left/joints');

  assert.deepEqual(calls[0].body, {
    tool: 'model',
    arguments: { topic: '/nvidia_desktop/hand/left/joints' },
  });
  assert.equal(calls[0].url, '/api/mcp/linkerhand-l20/call');
  assert.equal(renderer.urdf, '<robot name="stub"/>');
});

test('without a topic the model request keeps the old empty arguments', async () => {
  const { method, renderer, calls } = loadModelSandbox();
  await method.call(renderer, 'linkerhand-l20');

  assert.deepEqual(calls[0].body, { tool: 'model', arguments: {} });
  assert.equal(renderer.urdf, '<robot name="stub"/>');
});
