/**
 * x-options-action: what the config dialog does with a tool's reply.
 *
 * Run: node --test "agent-core/web/js/*.test.mjs"
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { parseToolReply, keptSelection, failureReply } from './config-options.js';

test('a reply arrives as MCP text content and is parsed', () => {
  const reply = { options: { printer: [{ const: 'QL', title: 'QL' }] }, help: { page_url: '已连接' } };
  assert.deepEqual(parseToolReply([{ type: 'text', text: JSON.stringify(reply) }]), reply);
});

test('text that is not JSON becomes the message, not an exception', () => {
  assert.deepEqual(parseToolReply([{ type: 'text', text: '前台电脑没响应' }]), { message: '前台电脑没响应' });
  assert.deepEqual(parseToolReply([{ type: 'text', text: '42' }]), { message: '42' });
  assert.deepEqual(parseToolReply(null), {});
});

test('the current pick survives new choices only while it is still offered', () => {
  const choices = [{ const: 'https://a/list' }, { const: 'https://b/list' }];
  assert.equal(keptSelection('https://b/list', choices), 'https://b/list');
  // A page the tool no longer reports is not silently kept as if it were.
  assert.equal(keptSelection('https://gone/list', choices), '');
  assert.equal(keptSelection('', choices), '');
});

test('a failed call shows its reason under the field that was asked about', () => {
  assert.deepEqual(failureReply('page_url', new Error('尚未配置：缺少 host')),
                   { help: { page_url: '尚未配置：缺少 host' } });
});
