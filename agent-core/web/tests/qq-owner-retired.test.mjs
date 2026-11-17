import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dirname = path.dirname(fileURLToPath(import.meta.url));
const js = filename => fs.readFileSync(path.join(dirname, '..', 'js', filename), 'utf8');

test('legacy QQ setup entry redirects to current QR onboarding, not retired pairing', () => {
  const legacy = js('channel-setup.js');
  assert.match(legacy, /export\s*\{\s*openQQWizard\s*\}\s*from\s*['"]\.\/qq-onboarding-v2\.js['"]/);
  assert.doesNotMatch(legacy, /qq\/pair|绑定管理员|确认管理员|全部 Agent Core 工具/);
});

test('active QQ setup entry uses QR flow and keeps other channels', () => {
  const entry = js('channel-setup-entry.js');
  assert.match(entry, /from ['"]\.\/qq-onboarding-v2\.js/);
  assert.match(entry, /select\.value !== 'qq'/);
  assert.doesNotMatch(js('qq-onboarding-v2.js'), /qq\/pair/);
});
