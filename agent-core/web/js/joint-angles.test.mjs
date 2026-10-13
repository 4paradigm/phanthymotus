import { test } from 'node:test';
import assert from 'node:assert/strict';
import { jointAngleRows, angleText, createJointAngleView } from './renderers/joint-angles.js';

test('degrees and radians come from the same raw q, sorted by motor index', () => {
  const rows = jointAngleRows({ joints: [
    { idx: 10, name: 'elbow_R_Joint', q: 0, group: '右臂', label: '肘' },
    { idx: 3, name: 'elbow_L_Joint', q: -Math.PI / 2, group: '左臂', label: '肘', q_deg: 999 },
  ] });
  assert.deepEqual(rows.map(row => row.idx), [3, 10]);
  assert.equal(rows[0].deg, -90);
  assert.equal(rows[0].rad, -Math.PI / 2);
  assert.equal(angleText(rows[0].rad), '-1.571');
  assert.equal(angleText(rows[1].rad), '0.000');
  assert.equal(angleText(-0.00001), '0.000');
});

test('invalid readings remain unknown rather than converting null to zero', () => {
  for (const q of [null, undefined, '0', true, NaN, Infinity]) {
    const row = jointAngleRows({ joints: [{ name: 'j', q }] })[0];
    assert.equal(row.rad, null);
    assert.equal(angleText(row.deg), '—');
  }
  assert.deepEqual(jointAngleRows({ status: { available: false }, joints: [{ q: 1 }] }), []);
  assert.deepEqual(jointAngleRows({ joints: null }), []);
});

class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.style = {}; this.attributes = {}; this._text = ''; this.open = false; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  set innerHTML(_) { throw new Error('Untrusted content must use textContent'); }
  appendChild(child) { this.children.push(child); child.parent = this; return child; }
  replaceChildren() { this.children = []; }
  setAttribute(key, value) { this.attributes[key] = value; }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(c => c !== this); }
}
const documentStub = { createElement: tag => new Element(tag) };
const nodes = (element, tag) => [element, ...element.children.flatMap(c => nodes(c, tag))].filter(e => e.tagName === tag);

test('table updates numbers in place, keeps expanded state and treats names as text', () => {
  globalThis.document = documentStub;
  const parent = new Element('div');
  const view = createJointAngleView(parent);
  try {
    const input = { joints: [{ idx: 3, name: '<img onerror=bad>', label: '肘', group: '左臂', q: 0 }] };
    view.update(input);
    view.element.open = true;
    const before = nodes(view.element, 'td');
    input.joints[0].q = -Math.PI / 2;
    view.update(input);
    const after = nodes(view.element, 'td');
    assert.equal(before[2], after[2]);
    assert.equal(after[2].textContent, '-90.000');
    assert.equal(after[3].textContent, '-1.571');
    assert.equal(after[3].title, String(-Math.PI / 2));
    assert.equal(after[1].title, '<img onerror=bad>');
    assert.equal(view.element.open, true);
    view.update({ status: { available: false }, joints: input.joints });
    assert.equal(nodes(view.element, 'td').length, 0);
    assert.match(nodes(view.element, 'summary')[0].textContent, /数据不可用/);
  } finally { view.dispose(); }
  assert.equal(parent.children.length, 0);
});

test('silence clears old numbers and separately mounted views do not share state', async () => {
  globalThis.document = documentStub;
  const first = createJointAngleView(new Element('div'), { staleMs: 5 });
  const second = createJointAngleView(new Element('div'), { staleMs: 10000 });
  try {
    first.update({ joints: [{ q: .1 }] });
    second.update({ joints: [{ q: .2 }] });
    await new Promise(resolve => setTimeout(resolve, 20));
    assert.equal(nodes(first.element, 'td').length, 0);
    assert.match(nodes(first.element, 'summary')[0].textContent, /数据已暂停/);
    assert.equal(nodes(second.element, 'td')[3].textContent, '0.200');
  } finally { first.dispose(); second.dispose(); }
});
