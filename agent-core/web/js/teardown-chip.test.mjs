/**
 * teardown-chip.test.mjs — 停止第 2 段那条芯片的状态与渲染。
 *
 * 这里守的是「界面说的话是真的」：
 *   - 还在跑 → 说在跑，不说「已完成」；
 *   - 有设备没停下来 → 说的是**后果**（可能仍在运行），不是完成率；
 *   - 没回话的项按失败算 —— 「没有结果」在这里的后果和报错一样；
 *   - 失败项排在最前并带重试，否则 6 张卡片里唯一要动手的那条要人去扫列表；
 *   - 全好才允许自动消失。有失败还自动消失，等于没报。
 *
 * Run: cd agent-core && node --test "web/js/*.test.mjs"
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  newTeardown, applyItem, applyDone, failedItems,
  chipLabel, autoHideMs, detailHtml,
} from './teardown-chip.js';

const CARDS = [
  { tool: 'remote_mic', mcp_id: 'agentcore', card_id: 'c-mic' },
  { tool: 'asr', mcp_id: 'mcp-1', card_id: 'c-asr' },
  { tool: 'tts', mcp_id: 'mcp-1', card_id: 'c-tts' },
];

test('刚开始时说的是进度，不是完成', () => {
  const s = newTeardown(CARDS);
  assert.deepEqual(chipLabel(s), { icon: '⏳', text: '设备收尾 0/3', tone: 'busy' });
  assert.equal(autoHideMs(s), 0, '还在跑就不能自己消失');
});

test('逐项标掉后进度前进', () => {
  const s = newTeardown(CARDS);
  applyItem(s, { card_id: 'c-mic', status: 'stopped' });
  applyItem(s, { card_id: 'c-asr', status: 'stopped' });
  assert.equal(chipLabel(s).text, '设备收尾 2/3');
});

test('全停完：说已收尾，并且可以自己消失', () => {
  const s = newTeardown(CARDS);
  CARDS.forEach((c) => applyItem(s, { card_id: c.card_id, status: 'stopped' }));
  applyDone(s);
  assert.deepEqual(chipLabel(s), { icon: '✓', text: '设备已收尾', tone: 'ok' });
  assert.ok(autoHideMs(s) > 0);
});

test('有失败：说的是「几台未停止」而不是完成率，而且不自动消失', () => {
  const s = newTeardown(CARDS);
  applyItem(s, { card_id: 'c-mic', status: 'stopped' });
  applyItem(s, { card_id: 'c-asr', status: 'stopped' });
  applyItem(s, { card_id: 'c-tts', status: 'error', message: '无响应（5s 死线）' });
  applyDone(s);
  assert.deepEqual(chipLabel(s), { icon: '⚠', text: '1 台设备未停止', tone: 'warn' });
  assert.equal(autoHideMs(s), 0, '有失败还自动消失，等于没报');
});

test('done 时还没回话的项按失败算', () => {
  const s = newTeardown(CARDS);
  applyItem(s, { card_id: 'c-mic', status: 'stopped' });
  applyDone(s);
  const failed = failedItems(s);
  assert.deepEqual(failed.map((f) => f.tool), ['asr', 'tts']);
  assert.equal(failed[0].message, '没有结果');
});

test('没有 card_id 时退回 tool+mcp_id 配对', () => {
  const s = newTeardown(CARDS);
  applyItem(s, { tool: 'asr', mcp_id: 'mcp-1', status: 'stopped' });
  assert.equal(s.items.find((i) => i.tool === 'asr').status, 'stopped');
});

test('不认识的项被忽略，不会凭空多出一行', () => {
  const s = newTeardown(CARDS);
  applyItem(s, { card_id: 'c-nope', status: 'stopped' });
  assert.equal(chipLabel(s).text, '设备收尾 0/3');
  assert.equal(s.items.length, 3);
});

test('失败项排最前，并且只有它带重试按钮', () => {
  const s = newTeardown(CARDS);
  applyItem(s, { card_id: 'c-mic', status: 'stopped' });
  applyItem(s, { card_id: 'c-asr', status: 'stopped' });
  applyItem(s, { card_id: 'c-tts', status: 'error', message: '无响应（5s 死线）' });
  applyDone(s);
  const html = detailHtml(s);
  assert.ok(html.indexOf('tts') < html.indexOf('remote_mic'), '失败项要排最前');
  assert.equal((html.match(/teardown-retry/g) || []).length, 1);
  assert.ok(html.includes('data-card-id="c-tts"'), '重试要带 card_id');
  assert.ok(html.includes('无响应（5s 死线）'));
});

test('工具名里的尖括号被转义 —— 它来自网络', () => {
  const s = newTeardown([{ tool: '<img src=x onerror=1>', mcp_id: 'm', card_id: 'c' }]);
  applyItem(s, { card_id: 'c', status: 'error', message: '<script>' });
  const html = detailHtml(s);
  assert.ok(!html.includes('<img'), html);
  assert.ok(!html.includes('<script>'), html);
  assert.ok(html.includes('&lt;img'));
});

test('空卡片列表不炸', () => {
  const s = newTeardown([]);
  assert.equal(chipLabel(s).text, '设备收尾 0/0');
  applyDone(s);
  assert.equal(chipLabel(s).tone, 'ok');
  assert.equal(detailHtml(s), '');
});
