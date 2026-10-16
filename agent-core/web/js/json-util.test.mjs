/**
 * Decoding the second layer of JSON in a bus payload.
 *
 * Producers double-encode: a field holds a JSON *string* rather than a JSON
 * value. Tool-call `arguments` are defined that way by the OpenAI schema and
 * arrive with non-ASCII escaped; a driver forwarding a response verbatim does
 * the same. Rendered as-is the panel showed source text — `你好` where
 * `你好` was meant, and backslash-escaped quotes around nested objects.
 *
 * The risk in decoding is over-reach, so most of this file pins what must be
 * left alone.
 *
 * Run: node --test "agent-core/web/js/*.test.mjs"
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { decodeNestedJson, plainValue, escapeHtml } from './renderers/json-util.js';

test('a field holding a JSON string is decoded into the value it encodes', () => {
  // exactly what a tool call puts on the bus: arguments as an escaped string
  const args = JSON.stringify({ text: '你好！很高兴' });
  const payload = { tool_calls: [{ name: 'channel_reply', args }] };
  const out = decodeNestedJson(payload);
  assert.deepEqual(out.tool_calls[0].args, { text: '你好！很高兴' });
});

test('escaped non-ASCII comes back as the characters it stands for', () => {
  // an ensure_ascii=True producer, or an OpenAI-compatible API
  const escaped = '{"text": "\\u4f60\\u597d"}';
  assert.deepEqual(decodeNestedJson({ a: escaped }), { a: { text: '你好' } });
});

test('values that merely look convertible are left alone', () => {
  const keep = {
    numericString: '123',
    prose: 'hello world',
    quoted: '"quoted prose"',
    topic: '/remote_control/image',
    real: 42,
    nothing: null,
  };
  assert.deepEqual(decodeNestedJson(keep), keep);
});

test('structural depth does not consume the unwrapping budget', () => {
  // The real /decision_core payload, at its real nesting:
  // decisions → round → tool_calls → call → args. An earlier version spent the
  // budget on plain structure and ran out one level short of the string, so the
  // card still showed escaped source text while a shallower fixture passed.
  const payload = {
    text: '',
    decisions: [{ round: 1, text: '', tool_calls: [{
      name: 'channel_reply',
      args: '{"text": "\\u4f60\\u597d\\uff01"}',
      result: '完成' }] }],
    source: 'collector',
  };
  const out = decodeNestedJson(payload);
  assert.deepEqual(out.decisions[0].tool_calls[0].args, { text: '你好！' });
});

test('a chain of strings encoding strings still terminates', () => {
  let deep = JSON.stringify({ x: 1 });
  for (let i = 0; i < 10; i++) deep = JSON.stringify({ n: deep });
  // The point is that it terminates and stays an object, not how far it got.
  assert.equal(typeof decodeNestedJson(deep), 'object');
});

test('malformed JSON stays as the text that was actually published', () => {
  // A producer emitting a broken argument string must still be visible as what
  // it sent, not swallowed.
  const broken = '{"text": "unterminated';
  assert.equal(decodeNestedJson({ a: broken }).a, broken);
});

test('arrays and nested objects are walked too', () => {
  const payload = { rows: [{ v: JSON.stringify({ deg: 1.5 }) }] };
  assert.deepEqual(decodeNestedJson(payload), { rows: [{ v: { deg: 1.5 } }] });
});

test('plainValue renders decoded data without quotes or escapes', () => {
  assert.equal(plainValue({ text: '你好' }), 'text=你好');
  assert.equal(plainValue([1, 2.5]), '1, 2.500');
  assert.equal(plainValue(null), '—');
  assert.equal(plainValue('已完成'), '已完成');
});

test('escapeHtml neutralises publisher-controlled markup', () => {
  // _formatJson assigns through innerHTML and every part of it comes off the bus
  assert.equal(escapeHtml('<img src=x onerror=alert(1)>'),
               '&lt;img src=x onerror=alert(1)&gt;');
  assert.equal(escapeHtml('a"b\'c&d'), 'a&quot;b&#39;c&amp;d');
});
