/** A card's config dialog owns both scopes: what it shows, and where it saves.
 *
 * Covers the two rules rather than the DOM — which keys land in the shared
 * group and which saved row each value is written to. A card's per-instance
 * numbers can depend on a shared setting (visual_depth's `cal_a`/`cal_b` are a
 * fit measured against whichever `backend` is running), and configuring them in
 * two dialogs is how r1_bj ended up with a YOLO fit selected under DepthART.
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


/** Mirrors the save split in sidebar.js: each row receives only its own keys. */
function splitByScope(props, values, sharedKeys) {
  const instanceValues = {}, sharedValues = {};
  for (const [k, v] of Object.entries(values)) {
    ((props[k] && props[k].scope !== 'instance') ? sharedValues : instanceValues)[k] = v;
  }
  const writes = [['instance', instanceValues]];
  if (sharedKeys.length) writes.push(['shared', sharedValues]);
  return writes;
}

test('each saved row receives only the keys it owns', () => {
  // A merged payload is rejected by plugins keeping one engine per process —
  // the same contract agent-core/src/tool_config.py applies on start.
  const writes = splitByScope(DEPTH_PROPS,
    { backend: 'depthart', fps: 2, cal_a: 1, cal_b: -0.31 }, ['backend']);
  assert.deepStrictEqual(writes, [
    ['instance', { fps: 2, cal_a: 1, cal_b: -0.31 }],
    ['shared', { backend: 'depthart' }],
  ]);
});

test('a tool with no shared fields never writes the tool-level row', () => {
  // Otherwise opening and saving a card would overwrite that row with {}.
  const props = { fps: { scope: 'instance', default: 2 } };
  const writes = splitByScope(props, { fps: 4 }, []);
  assert.deepStrictEqual(writes.map(w => w[0]), ['instance']);
});

test('an unknown key defaults to the instance row rather than the shared one', () => {
  // A stale saved key absent from the current schema must not be promoted to
  // tool-wide, where it would affect every other card.
  const writes = splitByScope(DEPTH_PROPS, { legacy_knob: 7 }, ['backend']);
  assert.deepStrictEqual(writes[0], ['instance', { legacy_knob: 7 }]);
  assert.deepStrictEqual(writes[1], ['shared', {}]);
});

/** The card is the only config entry once one exists. */
function sidebarOffersConfig(props, cardsOnCanvas) {
  if (cardsOnCanvas.length) return false;
  return Object.values(props).some(d => d.scope !== 'instance');
}

test('the sidebar steps aside once the tool has a card', () => {
  assert.strictEqual(sidebarOffersConfig(DEPTH_PROPS, [{ id: 'card-1' }]), false);
});

test('with no card the sidebar is still the only way in', () => {
  // A tool has to be configurable before you can usefully place a card —
  // ASR's url/key is the case.
  assert.strictEqual(sidebarOffersConfig(DEPTH_PROPS, []), true);
});

test('a tool with only instance fields never offers sidebar config', () => {
  assert.strictEqual(sidebarOffersConfig({ fps: { scope: 'instance' } }, []), false);
});

/** Mirrors sidebarOwnsConfig: step aside only if the card has a dialog. */
function ownsConfig(props, cards) {
  const v = Object.values(props);
  if (!v.some(d => d.scope !== 'instance')) return false;
  if (!v.some(d => d.scope === 'instance')) return true;
  return cards.length === 0;
}

test('a shared-only tool keeps its sidebar config even with a card', () => {
  // loco_servo: dry_run / rotate_only / require_standing are all shared, so
  // its card renders no config button — the sidebar is the only way in, and
  // hiding it put a chassis dry_run out of reach on R1.
  const loco = { dry_run: {}, rotate_only: {}, require_standing: {} };
  assert.strictEqual(ownsConfig(loco, [{ id: 'card-1' }]), true);
  assert.strictEqual(ownsConfig(loco, []), true);
});

test('a tool with both scopes still hands over once it has a card', () => {
  assert.strictEqual(ownsConfig(DEPTH_PROPS, [{ id: 'c' }]), false);
  assert.strictEqual(ownsConfig(DEPTH_PROPS, []), true);
});
