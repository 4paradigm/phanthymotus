const fs = require('fs');
const vm = require('vm');
const assert = require('assert').strict;
const path = require('path');
const source = fs.readFileSync(path.join(__dirname, '../web/js/sidebar.js'), 'utf8');
const start = source.indexOf('function _bindInstanceConfigVisibility(');
const end = source.indexOf('export async function openInstanceConfigModal', start);
const context = {};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context);
const controls = {
  calibration_preset: {value: 'no calibrate', addEventListener(_, fn) {this.change = fn;}},
};
const fields = [
  {dataset: {}, style: {}},
  {dataset: {showWhen: JSON.stringify({calibration_preset: 'manual set'})}, style: {}},
  {dataset: {showWhen: JSON.stringify({calibration_preset: ['manual set']})}, style: {}},
  {dataset: {showWhen: JSON.stringify({calibration_preset: 'manual set'})}, style: {}},
  {dataset: {hideWhen: JSON.stringify({calibration_preset: 'manual set'})}, style: {}},
];
const body = {
  querySelectorAll(selector) {return selector === '.tool-config-field' ? fields : Object.values(controls);},
  querySelector(selector) {return controls[selector.match(/data-key="([^"]+)"/)[1]];},
};
context._bindInstanceConfigVisibility(body);
assert.deepEqual(fields.map(f => f.style.display), ['', 'none', 'none', 'none', '']);
controls.calibration_preset.value = 'manual set';
controls.calibration_preset.change();
assert.deepEqual(fields.map(f => f.style.display), ['', '', '', '', 'none']);
controls.calibration_preset.value = 'no calibrate';
controls.calibration_preset.change();
assert.deepEqual(fields.map(f => f.style.display), ['', 'none', 'none', 'none', '']);
// Both renderers must honor schema defaults for friendly oneOf selectors.
assert.equal((source.match(/input.value = savedValues\[key\] \?\? def.default \?\? '';/g) || []).length, 2);
const instance = source.slice(end);
assert.ok(instance.includes('_bindInstanceConfigVisibility(bodyEl)'));
assert.ok(instance.includes("fieldWrapper.style.display === 'none'"));
console.log('instance visibility and model-default regression checks passed');
