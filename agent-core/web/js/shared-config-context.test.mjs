/** The tool-level settings a card's own config dialog shows, read-only.
 *
 * Covers the selection rule rather than the DOM: which keys are shown, and
 * what value each resolves to. A card's per-instance numbers can depend on a
 * shared setting — visual_depth's `cal_a`/`cal_b` are a fit measured against
 * whichever `backend` is running — and editing them with no way to see that
 * setting is how r1_bj ended up with a YOLO fit selected under DepthART.
 */
import { test } from 'node:test';
import assert from 'node:assert';

/** Mirrors the rule in sidebar.js: shared keys, resolved saved-then-default. */
function sharedContext(props, sharedSaved) {
  return Object.keys(props)
    .filter(k => props[k].scope !== 'instance')
    .map(k => {
      const current = sharedSaved[k] !== undefined ? sharedSaved[k] : props[k].default;
      return {
        key: k,
        label: props[k].title || k,
        value: current === undefined || current === null ? '(未设置)' : String(current),
      };
    });
}

const DEPTH_PROPS = {
  backend: { scope: 'shared', title: 'Depth model', default: 'yolo' },
  fps: { scope: 'instance', default: 2 },
  cal_a: { scope: 'instance', default: 1.0 },
  cal_b: { scope: 'instance', default: 0.0 },
};

test('only shared-scope keys are shown as context', () => {
  const rows = sharedContext(DEPTH_PROPS, {});
  assert.deepStrictEqual(rows.map(r => r.key), ['backend']);
});

test('a saved shared value wins over the schema default', () => {
  assert.strictEqual(sharedContext(DEPTH_PROPS, { backend: 'depthart' })[0].value, 'depthart');
  assert.strictEqual(sharedContext(DEPTH_PROPS, {})[0].value, 'yolo');
});

test('unset and empty-string are distinguishable', () => {
  // They are different states: unset falls back to whatever the service
  // decides, empty was chosen. Collapsing them hides which one is in force.
  const props = { url: { scope: 'shared', title: 'URL' } };
  assert.strictEqual(sharedContext(props, {})[0].value, '(未设置)');
  assert.strictEqual(sharedContext(props, { url: '' })[0].value, '');
});

test('a tool with no shared fields renders no context box', () => {
  const props = { fps: { scope: 'instance', default: 2 } };
  assert.strictEqual(sharedContext(props, {}).length, 0);
});

test('a key with no explicit scope counts as shared, matching the schema default', () => {
  const props = { provider: { title: 'Provider', default: 'x' } };
  assert.deepStrictEqual(sharedContext(props, {}).map(r => r.key), ['provider']);
});

test('false and 0 are rendered, not treated as unset', () => {
  const props = {
    dry_run: { scope: 'shared', title: 'Dry run', default: true },
    gain: { scope: 'shared', title: 'Gain', default: 1 },
  };
  const rows = sharedContext(props, { dry_run: false, gain: 0 });
  assert.deepStrictEqual(rows.map(r => r.value), ['false', '0']);
});
