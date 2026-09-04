/**
 * Deriving a card's output topic — the reason the ASR panel was not there at start.
 *
 * The monitor dashboard builds panels from the topics in the saved canvas layout.
 * A multiInstance tool's topic_out is derived by the driver from its input topic,
 * and only the canvas page ever asked for it, so whether the ASR panel existed
 * depended on someone having had the canvas open. These pin the resolution the
 * monitor now does for itself.
 *
 * Run: node --test "agent-core/web/js/*.test.mjs"
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { resolveDerivedTopics, inputTopicOf, selectPreviewTopic, topicOfPort } from './topic-derive.js';

// The layout that produced the report: mic → asr → tts, both derived cards saved
// with no topic because nothing had asked the driver yet.
const MIC = { id: 'card-mic', mcpId: 'mcp-1', toolName: 'remote_mic',
              topicOut: [{ topic: '/remote_control/mic', format: 'audio/pcm-16k' }] };
const ASR = { id: 'card-asr', mcpId: 'mcp-2', toolName: 'asr', topicOut: [] };
const TTS = { id: 'card-tts', mcpId: 'mcp-2', toolName: 'tts', topicOut: [] };
const CONNS = [
  { fromCardId: 'card-mic', fromPortIdx: '0', toCardId: 'card-asr', toPortIdx: '0', fromTopic: '/remote_control/mic' },
  { fromCardId: 'card-asr', fromPortIdx: '0', toCardId: 'card-tts', toPortIdx: '0', fromTopic: '' },
];

const clone = (o) => JSON.parse(JSON.stringify(o));

/** A driver that infers `input + '/' + tool`, like the real topic inference. */
function driver(log = []) {
  return async (url, init) => {
    const args = JSON.parse(init.body).arguments;
    log.push({ tool: JSON.parse(init.body).tool, input: args.input_topic });
    const inp = args.input_topic;
    const tool = JSON.parse(init.body).tool;
    return {
      json: async () => ({
        code: 200,
        data: { topic_out: [{ topic: inp ? `${inp}/${tool}` : '', format: 'data/json' }] },
      }),
    };
  };
}

test('a chain resolves end to end, so both panels exist', async () => {
  const cards = clone([MIC, ASR, TTS]);
  await resolveDerivedTopics(cards, clone(CONNS), { fetchImpl: driver() });
  assert.equal(cards[1].topicOut[0].topic, '/remote_control/mic/asr');
  assert.equal(cards[2].topicOut[0].topic, '/remote_control/mic/asr/tts');
});

test('the source card is left alone — its topic is static, not derived', async () => {
  const log = [];
  const cards = clone([MIC, ASR, TTS]);
  await resolveDerivedTopics(cards, clone(CONNS), { fetchImpl: driver(log) });
  assert.deepEqual(cards[0].topicOut, MIC.topicOut);
  assert.ok(!log.some(c => c.tool === 'remote_mic'), 'must not ask about a card that already has a topic');
});

test("an empty persisted fromTopic does not stop the downstream card", async () => {
  // The asr → tts connection was saved with fromTopic '' because ASR had not
  // resolved when the link was drawn. Reading the source card's topic instead of
  // that field is what lets TTS resolve at all.
  const cards = clone([MIC, ASR, TTS]);
  const conns = clone(CONNS);
  assert.equal(conns[1].fromTopic, '');
  await resolveDerivedTopics(cards, conns, { fetchImpl: driver() });
  assert.equal(cards[2].topicOut[0].topic, '/remote_control/mic/asr/tts');
});

test('a card already carrying a topic is not re-asked', async () => {
  const log = [];
  const cards = clone([MIC, ASR, TTS]);
  cards[1].topicOut = [{ topic: '/remote_control/mic/asr', format: 'data/json' }];
  await resolveDerivedTopics(cards, clone(CONNS), { fetchImpl: driver(log) });
  assert.deepEqual(log.map(c => c.tool), ['tts']);
});

test('a disconnected card is not asked — there is nothing to derive from', async () => {
  const log = [];
  const cards = clone([MIC, ASR, TTS]);
  await resolveDerivedTopics(cards, [], { fetchImpl: driver(log) });
  assert.deepEqual(log, []);
});

test('an offline driver costs one round, not maxRounds', async () => {
  let calls = 0;
  const dead = async () => { calls++; throw new Error('connection refused'); };
  const cards = clone([MIC, ASR, TTS]);
  await resolveDerivedTopics(cards, clone(CONNS), { fetchImpl: dead, maxRounds: 4 });
  assert.equal(calls, 1, `asked ${calls} times; a round that learns nothing must end it`);
  assert.deepEqual(cards[1].topicOut, []);
});

test('a driver that cannot infer is not retried forever', async () => {
  let calls = 0;
  const blank = async () => {
    calls++;
    return { json: async () => ({ code: 200, data: { topic_out: [{ topic: '', format: 'data/json' }] } }) };
  };
  const cards = clone([MIC, ASR, TTS]);
  await resolveDerivedTopics(cards, clone(CONNS), { fetchImpl: blank });
  assert.equal(calls, 1);
});

test('MCP content-item replies are parsed too', async () => {
  const wrapped = async () => ({
    json: async () => ({
      code: 200,
      data: [{ type: 'text', text: JSON.stringify({ topic_out: [{ topic: '/x/asr', format: 'data/json' }] }) }],
    }),
  });
  const cards = clone([MIC, ASR]);
  await resolveDerivedTopics(cards, [clone(CONNS[0])], { fetchImpl: wrapped });
  assert.equal(cards[1].topicOut[0].topic, '/x/asr');
});

test('malformed replies leave the layout untouched', async () => {
  const junk = async () => ({ json: async () => ({ code: 500, message: 'boom' }) });
  const cards = clone([MIC, ASR]);
  await resolveDerivedTopics(cards, [clone(CONNS[0])], { fetchImpl: junk });
  assert.deepEqual(cards[1].topicOut, []);
});

test('it asks with the resolved input topic, not the stale one', async () => {
  const log = [];
  const cards = clone([MIC, ASR, TTS]);
  // A leftover fromTopic from an older link that no longer reflects the graph.
  const conns = clone(CONNS);
  conns[1].fromTopic = '/remote_control/message';
  await resolveDerivedTopics(cards, conns, { fetchImpl: driver(log) });
  const ttsCall = log.find(c => c.tool === 'tts');
  assert.equal(ttsCall.input, '/remote_control/mic/asr');
  assert.equal(cards[2].topicOut[0].topic, '/remote_control/mic/asr/tts');
});

test('inputTopicOf and topicOfPort read the port that the connection names', () => {
  const two = { id: 'c', topicOut: [{ topic: '/a' }, { topic: '/b' }] };
  assert.equal(topicOfPort(two, 1), '/b');
  assert.equal(topicOfPort(two, 5), '/a', 'out of range falls back to the first port');
  assert.equal(topicOfPort({ }, 0), '');
  const conns = [{ fromCardId: 'c', fromPortIdx: '1', toCardId: 'd', toPortIdx: '0' }];
  assert.equal(inputTopicOf({ id: 'd' }, [two], conns), '/b');
});

test('the declared default preview wins without accepting a topic-less entry', () => {
  const topics = [
    { port: 'velocity', topic: '/velocity', format: 'data/json' },
    { port: 'costmap', topic: '/costmap', format: 'sensor/costmap' },
  ];
  assert.equal(
    selectPreviewTopic(topics, [{ port: 'costmap', default_preview: true }]),
    topics[1],
  );
  assert.equal(selectPreviewTopic([{ format: 'audio/pcm-16k' }]), null);
});

test('a source that is not on the canvas falls back to the persisted topic', async () => {
  // Last resort: nothing can derive this input, so the saved fromTopic is the
  // only evidence available. It is only consulted after every derivable card has
  // been derived, so it can no longer beat a real answer.
  const log = [];
  const orphanFed = { id: 'card-tts', mcpId: 'mcp-2', toolName: 'tts', topicOut: [] };
  const conns = [{ fromCardId: 'card-gone', fromPortIdx: '0', toCardId: 'card-tts',
                   toPortIdx: '0', fromTopic: '/remote_control/message' }];
  await resolveDerivedTopics([orphanFed], conns, { fetchImpl: driver(log) });
  assert.equal(orphanFed.topicOut[0].topic, '/remote_control/message/tts');
  assert.equal(log.length, 1);
});
