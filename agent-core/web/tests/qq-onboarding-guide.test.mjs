import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const wizardSource = readFileSync(new URL('../js/qq-onboarding-v2.js', import.meta.url), 'utf8');
const entrySource = readFileSync(new URL('../js/channel-setup-entry.js', import.meta.url), 'utf8');

test('QQ onboarding uses one normal add flow and displays a server-generated QR', () => {
  assert.match(wizardSource, /\/qq\/onboard\/start/);
  assert.match(wizardSource, /\/qq\/onboard\/status/);
  assert.match(wizardSource, /data-qq-qr/);
  assert.match(wizardSource, /扫码连接 QQ/);
  assert.doesNotMatch(wizardSource, /data-app-id|data-secret|AppID|AppSecret/);
  assert.doesNotMatch(wizardSource, /添加另一个 QQ 机器人|检测到已有机器人|选择已有机器人/);
  assert.doesNotMatch(wizardSource, /\/qq\/guided-setup|\/qq\/pair\/(start|confirm)/);
});

test('QR onboarding never treats a platform response as a successful connection by itself', () => {
  assert.match(wizardSource, /sessionStatus === 'connected'/);
  assert.match(wizardSource, /sessionStatus === 'configured'/);
  assert.match(wizardSource, /action_required/);
  assert.match(wizardSource, /data-qq-success-title/);
  assert.match(wizardSource, /data-qq-error/);
});

test('user can cancel, refresh an expired QR, and open the official QQ platform', () => {
  assert.match(wizardSource, /\/qq\/onboard\/cancel/);
  assert.match(wizardSource, /data-qq-refresh/);
  assert.match(wizardSource, /https:\/\/q\.qq\.com\/qqbot\/dashboard\//);
  assert.match(wizardSource, /rel="noopener noreferrer"/);
});

test('QR credentials are never read or rendered in the browser', () => {
  assert.doesNotMatch(wizardSource, /app_secret|client_secret|user_openid/);
  assert.doesNotMatch(wizardSource, /console\.(log|info|debug)/);
});

test('normal Add Channel flow opens QR only when QQ is selected', () => {
  assert.match(entrySource, /channel-add-btn/);
  assert.match(entrySource, /channel-form-platform/);
  assert.match(entrySource, /select\.value !== 'qq'/);
  assert.match(entrySource, /openQQWizard\(/);
});
