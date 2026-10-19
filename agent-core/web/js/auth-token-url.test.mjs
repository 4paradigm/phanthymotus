/**
 * 扫码带进来的 token —— 「手机接入」成不成立就在这一步。
 *
 * 页面从前只认 localStorage：`?token=` 对后端 API 有效，对页面无效，于是扫了码
 * 打开只会看到一个登录框，用户还得把访问码手抄一遍。二维码就白扫了。
 *
 * 另一半是抹掉。这个 token 能驱动电机，留在地址栏里它就会跟着截图、分享、
 * 书签一起走，还会进浏览历史。所以「收下」和「抹掉」是一件事，不是两件。
 *
 * Run: node --test "agent-core/web/js/*.test.mjs"
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

// auth.js 在模块顶层就要包 window.fetch，所以这些桩必须先于 import 装好。
const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};
globalThis.window = { fetch: async () => ({ ok: true }) };
globalThis.fetch = globalThis.window.fetch;

const nav = { replaceCount: 0, pushCount: 0 };
globalThis.location = {};
globalThis.history = {
  replaceState: (_s, _t, url) => { nav.replaceCount++; _setUrl(url); },
  pushState: () => { nav.pushCount++; },
};

function _setUrl(href) {
  const u = new URL(href, 'https://10.100.121.14:15678');
  globalThis.location.href = u.href;
  globalThis.location.pathname = u.pathname;
  globalThis.location.search = u.search;
  globalThis.location.hash = u.hash;
}

const { consumeTokenFromUrl, getToken, clearToken } = await import('./auth.js');

function visit(url) {
  store.clear();
  nav.replaceCount = 0;
  nav.pushCount = 0;
  _setUrl(url);
}

const currentUrl = () => location.pathname + location.search + location.hash;

test('查询串里的 token 被收下，并从地址栏抹掉', () => {
  visit('/?token=SCANNED');
  assert.equal(consumeTokenFromUrl(), true);
  assert.equal(getToken(), 'SCANNED');
  assert.equal(currentUrl(), '/', 'token 还留在地址栏里');
});

test('抹掉走 replaceState —— 不能给这个 token 留一条历史记录', () => {
  visit('/?token=SCANNED');
  consumeTokenFromUrl();
  assert.equal(nav.replaceCount, 1);
  assert.equal(nav.pushCount, 0);
});

test('只删 token，同行的其它参数留着', () => {
  visit('/?mode=monitor&token=SCANNED&card=3');
  consumeTokenFromUrl();
  assert.equal(getToken(), 'SCANNED');
  assert.equal(currentUrl(), '/?mode=monitor&card=3');
});

test('fragment 形式也认 —— 有些扫码器会把查询串洗掉', () => {
  visit('/#token=HASHED');
  assert.equal(consumeTokenFromUrl(), true);
  assert.equal(getToken(), 'HASHED');
  assert.equal(currentUrl(), '/');
});

test('token 里的特殊字符按百分号编码还原', () => {
  const token = 'a+b/c=d_e-f';
  visit(`/?token=${encodeURIComponent(token)}`);
  consumeTokenFromUrl();
  assert.equal(getToken(), token);
});

test('没有 token 时什么都不做 —— 不清掉已经存着的那个', () => {
  visit('/');
  globalThis.localStorage.setItem('phanthy_access_token', 'EXISTING');
  assert.equal(consumeTokenFromUrl(), false);
  assert.equal(getToken(), 'EXISTING', '已有的 token 被误清了');
  assert.equal(nav.replaceCount, 0, '没东西要抹，就不该动地址栏');
});

test('URL 里的 token 覆盖旧的 —— 换了机器扫码要能顶掉上一台的', () => {
  visit('/?token=NEW');
  globalThis.localStorage.setItem('phanthy_access_token', 'OLD');
  consumeTokenFromUrl();
  assert.equal(getToken(), 'NEW');
});

test('clearToken 之后确实没有了', () => {
  visit('/?token=SCANNED');
  consumeTokenFromUrl();
  clearToken();
  assert.equal(getToken(), '');
});
