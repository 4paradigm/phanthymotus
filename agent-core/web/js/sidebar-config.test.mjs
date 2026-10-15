import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('./sidebar.js', import.meta.url), 'utf8');
const schema = { default: 'normal', oneOf: [
  { const: 'normal', title: '普通环境' },
  { const: 'noisy', title: '嘈杂环境' },
] };

// Run each modal's actual oneOf rendering branch with a minimal select DOM.
function render(modal, savedValues, def = schema) {
  const start = source.indexOf('if (def.oneOf && Array.isArray(def.oneOf)) {',
    source.indexOf(`export async function ${modal}(`));
  const end = source.indexOf('    } else if (def.enum', start);
  const document = { createElement(tag) {
    const element = { children: [], dataset: {}, selected: false,
      appendChild(child) { this.children.push(child); } };
    let value = '';
    Object.defineProperty(element, 'value', {
      get() { return tag === 'select'
        ? this.children.find(child => child.selected)?.value ?? this.children[0]?.value ?? ''
        : value; },
      set(next) {
        if (tag === 'select') this.children.forEach(child => { child.selected = child.value === String(next); });
        else value = String(next);
      },
    });
    return element;
  } };
  return vm.runInNewContext(`let input; ${source.slice(start, end)} } input;`,
    { def, key: 'sensitivity', savedValues, document });
}

for (const modal of ['openToolConfigModal', 'openInstanceConfigModal']) {
  for (const [label, saved, expected] of [
    ['missing', {}, 'normal'],
    ['saved choice', { sensitivity: 'noisy' }, 'noisy'],
    ['old choice', { sensitivity: 'low' }, 'normal'],
    ['empty choice', { sensitivity: '' }, 'normal'],
  ]) {
    test(`${modal}: ${label} selects ${expected}`, () => {
      const select = render(modal, saved);
      assert.equal(select.value, expected);
      assert.equal(select.children.find(option => option.value === expected).selected, true);
    });
  }
  test(`${modal}: no default keeps the placeholder`, () => {
    assert.equal(render(modal, {}, { oneOf: schema.oneOf }).value, '');
  });
}
