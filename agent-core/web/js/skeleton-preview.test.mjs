import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

function sourceOf(file, signature, end = '\n}') {
  const source = readFileSync(new URL(file, import.meta.url), 'utf8').replaceAll('\r\n', '\n');
  const start = source.indexOf(signature);
  assert.notEqual(start, -1, signature);
  const stop = source.indexOf(end + '\n', start);
  assert.notEqual(stop, -1, signature);
  return source.slice(start, stop + end.length);
}

const openTopic = sourceOf('./canvas.js', 'async function _openTopicDetailFor(');
const showDetail = sourceOf('./detail-panel.js', 'function showTopicDetail(');
const loadModelSource = sourceOf('./renderers/skeleton.js', 'async _loadModel(', '\n  },');
const as2wTopic = '/nvidia_desktop/state/joints';
const cachedTopic = '/as2w_cached/state/joints';
const skeleton = 'sensor/skeleton';

async function openPreview(mcpId, liveTopic) {
  const source = {
    querySelectorAll: () => [{ dataset: { topic: cachedTopic, format: skeleton } }],
  };
  const elements = {
    'detail-title': {},
    'detail-subtitle': {},
    'detail-body': { style: {}, appendChild() {} },
  };
  const mounted = [];
  const connected = [];
  const open = runInNewContext(openTopic + '\n' + showDetail + '\n_openTopicDetailFor', {
    _cards: [{ id: 'joint-card', el: source, toolName: 'joints' }],
    _allMcps: [],
    _inputTopicsFor: () => [],
    inputArgs: () => ({}),
    _parseMcpCallResult: reply => reply.data,
    _panel: { classList: { remove() {} } },
    _cleanup() {},
    _connect(topic, format) { connected.push([topic, format]); },
    _setStatus() {},
    _CENTERED: '',
    _session: 1,
    openDetailPanelMobile() {},
    document: {
      getElementById: id => elements[id],
      createElement: () => ({ style: {} }),
    },
    RENDERERS: [{
      canRender: hint => hint === skeleton,
      mount(container, id) { mounted.push([container, id]); },
    }],
    TextRenderer: null,
    ActivityRenderer: null,
    fetch: async (url, request) => {
      assert.equal(url, '/api/mcp/' + encodeURIComponent(mcpId) + '/call');
      assert.equal(JSON.parse(request.body).tool, 'joints');
      return { json: async () => ({
        data: liveTopic ? { topic_out: [{ topic: liveTopic, format: skeleton }] } : null,
      }) };
    },
  });
  await open(source, mcpId, []);
  assert.deepEqual(mounted, [[elements['detail-body'], mcpId]]);
  assert.deepEqual(connected, [[liveTopic || cachedTopic, skeleton]]);
}

test('AS2-W skeleton preview passes the card owner to the renderer for a live topic', async () => {
  await openPreview('unitree-as2w', as2wTopic);
});

test('skeleton preview keeps the card owner when it falls back to a cached topic', async () => {
  await openPreview('another-robot', null);
});

test('an unmounted skeleton preview ignores both successful and failed model replies', async () => {
  for (const failed of [false, true]) {
    let resolveFetch, rejectFetch;
    const response = new Promise((resolve, reject) => {
      resolveFetch = resolve;
      rejectFetch = reject;
    });
    const loadModel = runInNewContext('({' + loadModelSource + '})._loadModel', {
      fetch: () => response,
      console: { warn: () => assert.fail('a closed preview should ignore the reply') },
    });
    const renderer = {
      _scene: {},
      _parseAndBuild: () => assert.fail('a closed preview should not build a URDF'),
      _buildFallbackSkeleton: () => assert.fail('a closed preview should not build a fallback'),
    };
    const loading = loadModel.call(renderer, 'unitree-as2w');
    renderer._scene = null;
    if (failed) rejectFetch(new Error('model request failed'));
    else resolveFetch({ json: async () => ({ data: [{ text: '{"urdf":"<robot/>"}' }] }) });
    await loading;
  }
});
