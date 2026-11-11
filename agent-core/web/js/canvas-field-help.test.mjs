import test from 'node:test';
import assert from 'node:assert/strict';

import { renderCanvasFieldHelp } from './canvas-field-help.js';

test('omits help for absent, empty, and blank descriptions', () => {
  for (const description of [undefined, null, '', ' \n\t ']) {
    assert.deepEqual(renderCanvasFieldHelp(description, 'field-1'), {
      inputAttributes: '',
      helpMarkup: ''
    });
  }
});

test('links the input to full help text and escapes markup and quotes', () => {
  const description = 'Use <sensor> & "quotes" and \'apostrophes\'.\nKeep this newline.';
  const result = renderCanvasFieldHelp(description, 'sensor-1');

  assert.equal(result.inputAttributes, ' aria-describedby="sensor-1-help"');
  assert.equal(
    result.helpMarkup,
    '<div class="canvas-field-help" id="sensor-1-help">Use &lt;sensor&gt; &amp; &quot;quotes&quot; and &#39;apostrophes&#39;.\nKeep this newline.</div>'
  );
  assert.ok(result.helpMarkup.includes('\nKeep this newline.'));
});

test('keeps long CJK descriptions without truncation', () => {
  const description = '传感器参数说明'.repeat(80);
  const result = renderCanvasFieldHelp(description, 'sensor-long');

  assert.equal(result.inputAttributes, ' aria-describedby="sensor-long-help"');
  assert.ok(result.helpMarkup.includes(description));
  assert.ok(result.helpMarkup.endsWith('</div>'));
});

test('uses distinct deterministic IDs per field and escapes quoted attribute values', () => {
  const first = renderCanvasFieldHelp('First help', 'field-a');
  const second = renderCanvasFieldHelp('Second help', 'field-b');

  assert.notEqual(first.inputAttributes, second.inputAttributes);
  assert.match(first.inputAttributes, /field-a-help/);
  assert.match(second.inputAttributes, /field-b-help/);

  const escaped = renderCanvasFieldHelp('Help', 'id"<&');
  assert.equal(
    escaped.inputAttributes,
    ' aria-describedby="id&quot;&lt;&amp;-help"'
  );
  assert.equal(
    escaped.helpMarkup,
    '<div class="canvas-field-help" id="id&quot;&lt;&amp;-help">Help</div>'
  );
});
