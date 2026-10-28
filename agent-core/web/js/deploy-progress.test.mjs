/**
 * Agent Core 自升级的成功判定。
 *
 * 快机器可能在两次 2 秒轮询之间完成 stop/start；浏览器没见过断线也必须能凭
 * 新容器报告的目标 tag 收尾，不能永远停在 90%。
 *
 * Run: cd agent-core && node --test "web/js/*.test.mjs"
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

// deploy-progress.js 在模块加载时注册 DOMContentLoaded；本测试只测纯判定函数。
globalThis.document = {
  readyState: 'loading',
  addEventListener: () => {},
};

const { _coreReachedTarget } = await import('./deploy-progress.js');

test('没观察到断线也能凭目标 tag 判定升级成功', () => {
  assert.equal(
    _coreReachedTarget(
      'ccr.example.com/phanthy-motus/core:release.261023.3a3567a',
      'ccr.example.com/phanthy-motus/core:release.261023.3a3567a',
    ),
    true,
  );
});

test('仍在旧版本时不能误报成功', () => {
  assert.equal(
    _coreReachedTarget(
      'ccr.example.com/phanthy-motus/core:release.261022.old',
      'ccr.example.com/phanthy-motus/core:release.261023.3a3567a',
    ),
    false,
  );
});

test('registry 端口不是镜像 tag', () => {
  assert.equal(
    _coreReachedTarget(
      'registry.example.com:5000/ns/core:release.261023.3a3567a',
      'registry.example.com:5000/ns/core:release.261023.3a3567a',
    ),
    true,
  );
  assert.equal(_coreReachedTarget('registry.example.com:5000/ns/core', ''), false);
});
