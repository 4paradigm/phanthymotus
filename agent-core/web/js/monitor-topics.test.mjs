import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { inputBindingsOf, syncConnectionTopics } from './topic-derive.js';

const source = readFileSync(new URL('./monitor-dashboard.js', import.meta.url), 'utf8');
function refresh(cards, connections, mcps, resolve = () => {}) {
  const context = vm.createContext({ inputBindingsOf });
  for (const name of ['_refreshStaticTopics', '_refreshInputTopics']) {
    const start = source.indexOf(`function ${name}(`);
    assert.notEqual(start, -1);
    const end = source.indexOf('\nfunction ', start + 1);
    vm.runInContext(source.slice(start, end), context);
  }
  context._refreshStaticTopics(cards, mcps);
  resolve();
  syncConnectionTopics(cards, connections);
  context._refreshInputTopics(cards, connections);
}

test('monitor replaces old sensor output and connected input with the current topic', () => {
  const cards = [
    { id: 'sensor', mcpId: 'driver', toolName: 'sound_direction',
      topicOut: [{ topic: '/robot/sound_direction/events', format: 'sensor/sound-direction' }] },
    { id: 'asr', mcpId: 'perception', toolName: 'asr',
      topicIn: [{ topic: '/robot/sound_direction/events', format: 'sensor/sound-direction' }],
      topicOut: [{ topic: '/mic/asr', format: 'data/json' }] },
  ];
  const connections = [{ fromCardId: 'sensor', toCardId: 'asr',
    fromPortIdx: '0', toPortIdx: '0', fromTopic: '/robot/sound_direction/events' }];
  const mcps = [{ id: 'driver', tools: [{ name: 'sound_direction',
    topic_out: [{ topic: '/robot/sound_direction', format: 'sensor/sound-direction' }] }] },
  { id: 'perception', tools: [{ name: 'asr', multiInstance: true,
    topic_out: [{ format: 'data/json' }] }] }];
  refresh(cards, connections, mcps);
  syncConnectionTopics(cards, connections);
  assert.equal(cards[0].topicOut[0].topic, '/robot/sound_direction');
  assert.equal(cards[1].topicIn[0].topic, '/robot/sound_direction');
  assert.equal(cards[1].topicOut[0].topic, '/mic/asr');
  assert.equal(connections[0].fromTopic, '/robot/sound_direction');
  const topics = new Set(cards.flatMap(c => [...(c.topicOut || []), ...(c.topicIn || [])])
    .map(port => port.topic));
  assert.equal(topics.has('/robot/sound_direction/events'), false);
});

test('offline or unresolved declarations preserve saved topics', () => {
  const cards = [{ id: 'camera', mcpId: 'driver', toolName: 'camera',
    topicOut: [{ topic: '/camera/image', format: 'image/jpeg' }] }];
  const before = JSON.stringify(cards);
  refresh(cards, [], []);
  assert.equal(JSON.stringify(cards), before);
  refresh(cards, [], [{ id: 'driver', tools: [{ name: 'camera', topic_out: [{ format: 'image/jpeg' }] }] }]);
  assert.equal(JSON.stringify(cards), before);
});

test('downstream input follows a derived output resolved after static refresh', () => {
  const cards = [
    { id: 'asr', mcpId: 'perception', toolName: 'asr',
      topicOut: [{ topic: '/old/mic/asr', format: 'data/json' }] },
    { id: 'tts', mcpId: 'perception', toolName: 'tts',
      topicIn: [{ topic: '/old/mic/asr', format: 'data/json' }] },
  ];
  const connections = [{ fromCardId: 'asr', toCardId: 'tts', fromPortIdx: '0',
    toPortIdx: '0', fromTopic: '/old/mic/asr' }];
  const mcps = [{ id: 'perception', tools: [
    { name: 'asr', multiInstance: true }, { name: 'tts', multiInstance: true }] }];
  refresh(cards, connections, mcps, () => {
    cards[0].topicOut = [{ topic: '/new/mic/asr', format: 'data/json' }];
  });
  assert.equal(cards[1].topicIn[0].topic, '/new/mic/asr');
  assert.equal(connections[0].fromTopic, '/new/mic/asr');
});

test('a transport events suffix is not the window name', () => {
  const context = vm.createContext({});
  const start = source.indexOf('function _cardShortName(');
  assert.notEqual(start, -1);
  const end = source.indexOf('\nfunction ', start + 1);
  vm.runInContext(source.slice(start, end), context);
  assert.equal(context._cardShortName('/ubuntu/sound_direction/events'), 'sound_direction');
  assert.equal(context._cardShortName('/ubuntu/sound_direction'), 'sound_direction');
  assert.equal(context._cardShortName('/cam/image'), 'image');
});
