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
const onDataSource = sourceOf('./renderers/skeleton.js', 'onData(buffer, fmt)', '\n  },');
const as2wTopic = '/nvidia_desktop/state/joints';
const cachedTopic = '/as2w_cached/state/joints';
const skeleton = 'sensor/skeleton';

async function openPreview(mcpId, liveTopic, format = skeleton) {
  const source = {
    querySelectorAll: () => [{ dataset: { topic: cachedTopic, format } }],
  };
  const elements = {
    'detail-title': {},
    'detail-subtitle': {},
    'detail-body': { style: {}, appendChild() {} },
  };
  const mounted = [];
  const connected = [];
  const renderer = {
    canRender: hint => hint === format,
    mount(container, id) { mounted.push([container, id]); },
  };
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
    RENDERERS: [renderer],
    SkeletonRenderer: format === skeleton ? renderer : null,
    TextRenderer: null,
    ActivityRenderer: null,
    fetch: async (url, request) => {
      assert.equal(url, '/api/mcp/' + encodeURIComponent(mcpId) + '/call');
      assert.equal(JSON.parse(request.body).tool, 'joints');
      return { json: async () => ({
        data: liveTopic ? { topic_out: [{ topic: liveTopic, format }] } : null,
      }) };
    },
  });
  await open(source, mcpId, []);
  assert.deepEqual(mounted, [[elements['detail-body'], format === skeleton ? mcpId : 'detail']]);
  assert.deepEqual(connected, [[liveTopic || cachedTopic, format]]);
}

test('AS2-W skeleton preview passes the card owner to the renderer for a live topic', async () => {
  await openPreview('unitree-as2w', as2wTopic);
});

test('skeleton preview keeps the card owner when it falls back to a cached topic', async () => {
  await openPreview('another-robot', null);
});

test('JSON streams keep the existing detail renderer context', async () => {
  await openPreview('another-robot', '/other_namespace/joint_state', 'data/json');
});

test('a joint snapshot received before the URDF is applied once the model loads', async () => {
  let sendModel;
  const response = new Promise(resolve => { sendModel = resolve; });
  const renderer = runInNewContext('({' + loadModelSource + onDataSource + '})', {
    fetch: () => response,
    TextDecoder,
    console: { warn: () => assert.fail('the URDF request should succeed') },
  });
  renderer._scene = {};
  const rotations = [];
  renderer._parseAndBuild = () => {
    renderer._joints = {
      hinge: {
        userData: { axis: {} },
        quaternion: { setFromAxisAngle(_axis, angle) { rotations.push(angle); } },
      },
    };
    renderer._loaded = true;
  };
  const loading = renderer._loadModel('another-robot');
  const encoder = new TextEncoder();
  for (const q of [0.25, 0.5]) {
    renderer.onData(encoder.encode(JSON.stringify({ joints: [{ name: 'hinge', q }] })));
  }
  assert.deepEqual(rotations, []);
  sendModel({ json: async () => ({ data: [{ text: '{"urdf":"<robot/>"}' }] }) });
  await loading;
  assert.deepEqual(rotations, [0.5]);
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
