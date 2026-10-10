/**
 * activity-summary.test.mjs — 活动流每种事件渲染成哪一行。
 *
 * 这个文件存在的原因是 activity-log.js 本身在 `node --test` 下 import 不进来（它
 * 牵进 mobile.js，再牵进整棵 renderer 树和 three.js）。摘要函数因此被拆到
 * activity-summary.js —— 一个不 import 任何东西的文件 —— 这样每种行都测得到。
 * 这里错一个分支，在 review 里看不出来，到机器人上就是一行原始 JSON。
 *
 * Run: cd agent-core && node --test "web/js/*.test.mjs"
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { summarizeEvent } from './activity-summary.js';

test('背景语音带上说话人、语种和情绪', () => {
  assert.equal(
    summarizeEvent({ type: 'asr_background', payload: {
      text: '今天天气不错', speaker_name: '云强', lang: 'zh', emotion: 'HAPPY' } }),
    '👂 "今天天气不错" · 云强 · zh · HAPPY');
});

test('背景语音没有附加字段时只有那句话', () => {
  assert.equal(
    summarizeEvent({ type: 'asr_background', payload: { text: 'The.' } }),
    '👂 "The."');
});

test('空的说话人字段不占一行日志的宽度', () => {
  assert.equal(
    summarizeEvent({ type: 'asr_background', payload: {
      text: '嗯', speaker_name: '', speaker_id: null, lang: 'zh' } }),
    '👂 "嗯" · zh');
});

test('背景语音和前台识别结果是两种行', () => {
  // 同样一句话，前台那条已经由 trigger / asr_result 渲染过 —— 两者不能长得一样，
  // 否则「机器人在跟谁说话」这件事在日志里就看不出来了。
  const bg = summarizeEvent({ type: 'asr_background', payload: { text: '你好' } });
  const fg = summarizeEvent({ type: 'asr_result', payload: { text: '你好' } });
  assert.notEqual(bg, fg);
  assert.ok(bg.startsWith('👂'));
});

test('缺 payload 不抛', () => {
  assert.equal(summarizeEvent({ type: 'asr_background' }), '👂 ""');
});

test('未知类型回落到截断后的 JSON', () => {
  const out = summarizeEvent({ type: 'something_new', payload: { a: 1 } });
  assert.equal(out, '{"a":1}');
});

test('截断是从中间切的，两头都留着', () => {
  const long = 'x'.repeat(500);
  const out = summarizeEvent({ type: 'mcp_result', payload: { result: long } });
  assert.ok(out.includes('…（省略'));
  assert.ok(out.endsWith('x"'), out.slice(-20));
});
