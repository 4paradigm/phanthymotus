import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('执行卡片按钮传入实际 card id', async () => {
  const source = readFileSync(new URL('./canvas.js', import.meta.url), 'utf8');
  const start = source.indexOf("    const execBtn = el.querySelector('.canvas-exec-btn');");
  const end = source.indexOf("    const viewBtn =", start);
  assert.ok(start >= 0 && end > start);
  let click;
  let called;
  const button = { addEventListener: (name, fn) => { assert.equal(name, 'click'); click = fn; } };
  const el = { querySelector: () => button };
  new Function('el', 'mcpId', 'toolName', 'id', '_ensureEdit', '_executeCard', source.slice(start, end))(
    el, 'channel', 'fanshi_message_send', 'card-123', async () => true,
    async (...args) => { called = args; },
  );
  await click({ stopPropagation() {} });
  assert.deepEqual(called, [el, 'channel', 'fanshi_message_send', 'card-123']);
});
