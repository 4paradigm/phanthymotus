/**
 * handoff.js — 「手机接入」：把这个控制台交到手上。
 *
 * 渲染一张二维码，内容是本机的局域网地址加上访问 token，手机扫一下就进来了，
 * 不用记 IP、不用手输 token。落在「我的」页里，桌面 modal 和移动端整页面板
 * 共用同一份 HTML，跟 account.js 的做法一致。
 *
 * 三件事值得先说清楚：
 *
 * 1. **地址必须问服务端要，不能用 `location.host`。** 机器上本地开的浏览器看到
 *    的是 `localhost`，SSH 端口转发看到的是 `127.0.0.1` —— 编进二维码，手机拨
 *    不通。`/api/network/reachable` 给的是网卡上的地址。
 *
 * 2. **二维码是凭证，不是装饰。** 扫进去的人拿到的是满权限的控制台，能驱动电机。
 *    所以默认盖住，点一下才显示，并且到时自动收起 —— 控制台常年开在大屏上，
 *    也常被投屏。
 *
 * 3. **扫码那一端要能认这个 token。** 见 auth.js 的 `consumeTokenFromUrl()`：
 *    页面从前进不认 URL 里的 token，只认 localStorage，所以扫码只会开出一个
 *    登录框来。
 */

import { getToken } from './auth.js';
import { loadLocalAddresses } from './local-address.js';
import { qrSvg } from './qrcode.js';
import { showToast } from './toast.js';

/** 显示后多久自动收起。够扫，不够被人从会议室另一头拍下来慢慢试。 */
const REVEAL_MS = 90_000;

let _addresses = null;     // null = 还没问过；[] = 问过了但一个都没有
let _selected = '';        // 选中的 IP
let _revealUntil = 0;      // 时间戳；0 表示盖着
let _timer = null;

// ── 数据 ────────────────────────────────────────────────────────────────────

/**
 * 拉本机可达地址。列表由 local-address.js 缓存，和插件配置页里的二维码字段
 * 共用；选中项（`_selected`）留在这里各自持有。
 */
export async function loadHandoffAddresses() {
  if (_addresses) return;
  _addresses = await loadLocalAddresses();
  if (!_selected) _selected = _addresses[0]?.ip || '';
}

/** 扫码打开的完整地址。端口与协议沿用当前这一个，只换主机名。 */
function _handoffUrl() {
  if (!_selected) return '';
  const url = new URL(location.origin);
  url.hostname = _selected;
  url.pathname = '/';
  const token = getToken();
  if (token) url.searchParams.set('token', token);
  return url.toString();
}

// ── 渲染 ────────────────────────────────────────────────────────────────────

/**
 * 「手机接入」那一段的 HTML。由 account.js 拼进它自己的 innerHTML，
 * 写完之后必须调一次 `mountHandoff()` 把倒计时接回去。
 */
export function handoffMarkup() {
  // 还没问到地址：整段先不画。画出来只会是一个空白方块加一个按不出东西的
  // 「显示二维码」—— 一个点了没反应的按钮比晚半拍出现的区块糟得多。
  if (_addresses === null) return '';

  if (!_addresses.length) {
    return `
      <div class="account-section-label">手机接入</div>
      <div class="account-card handoff-card handoff-card-empty">
        <p class="handoff-empty-text">问不到这台机器的局域网地址。确认机器已连上网络后重开这一页。</p>
      </div>`;
  }

  const url = _handoffUrl();
  const gated = !!getToken();                  // 没开鉴权就没有秘密，不必盖
  const revealed = !gated || Date.now() < _revealUntil;
  const remain = Math.max(0, _revealUntil - Date.now());

  return `
    <div class="account-section-label">手机接入</div>
    <div class="account-card handoff-card">
      <div class="handoff-stub">
        <!-- 不要给这里任何 id：account.js 把同一份 HTML 同时写进桌面 modal 和
             移动端面板两个容器，id 会在文档里出现两份。 -->
        <div class="handoff-qr ${revealed ? '' : 'is-covered'}">
          ${url ? qrTile(url) : ''}
          ${revealed ? '' : `
            <button class="handoff-reveal" data-handoff="reveal">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"
                   stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/>
                <circle cx="12" cy="12" r="3"/>
              </svg>
              显示二维码
            </button>`}
          ${revealed && gated ? `
            <span class="handoff-countdown" style="animation-duration:${REVEAL_MS}ms;
                  animation-delay:-${REVEAL_MS - remain}ms"></span>` : ''}
        </div>
      </div>
      <div class="handoff-detail">
        <code class="handoff-url" title="${_esc(_displayUrl(url, true))}">${_esc(_displayUrl(url))}</code>
        <p class="handoff-where">${_esc(_whereText())}</p>
        ${_renderAddressPicker()}
        <div class="handoff-actions">
          <button class="account-copy-btn" data-handoff="copy">复制链接</button>
          ${revealed && gated
            ? `<button class="account-link handoff-hide" data-handoff="hide">收起</button>`
            : ''}
        </div>
      </div>
    </div>
    <p class="account-hint">${gated
      ? '扫码后手机直接进这个控制台，不用再输访问码。二维码带着访问码，显示 90 秒后自动收起。'
        + '手机上浏览器会提示证书不受信任 —— 那是机器人自签的证书，选「继续访问」。'
      : '这台机器没有设访问码，扫到这个二维码的人都能打开控制台。'}</p>`;
}

/** 二维码本体。内容太长（自定义的超长访问码）时不画，留「复制链接」这条路。 */
function qrTile(url) {
  try {
    // margin 4 是规范要求的静区宽度，别为了把码画大一点而省它
    return qrSvg(url, { margin: 4, color: '#1C1917', background: '#FFFFFF' });
  } catch {
    return '<span class="handoff-qr-fallback">链接太长，画不成二维码</span>';
  }
}

/**
 * 地址里最该被人眼读到的是主机和端口，token 那一长串没人会照着抄。
 *
 * tooltip 里也不放 token：盖住二维码是为了挡住「被动看到」这条路，
 * 那么鼠标一悬停就把访问码明文摊出来，等于把刚关上的门又开了。
 */
function _displayUrl(url, withScheme = false) {
  if (!url) return '';
  try {
    const u = new URL(url);
    return withScheme ? `${u.protocol}//${u.host}/` : u.host;
  } catch {
    return url;
  }
}

/**
 * 网卡名只在没有选择器可说的时候才由这行来说 —— 有选择器时它已经写在
 * 按钮上了，两处写同一件事是噪音。
 */
function _whereText() {
  const many = (_addresses?.length || 0) > 1;
  if (many) return '手机要和这台机器连同一个网络';
  const addr = _addresses?.[0];
  const via = _addrLabel(addr);
  return via ? `${via} · 手机要连同一个网络` : '手机要和这台机器连同一个网络';
}

/** 网卡名比 IP 好认（哪个是 WiFi、哪个是接本体的网线）；认不出就退回 IP。 */
function _addrLabel(addr) {
  if (!addr) return '';
  return addr.device || addr.ip;
}

/** 只有一个地址就不画选择器 —— 一个选项的选择器是噪音。 */
function _renderAddressPicker() {
  if (!_addresses || _addresses.length < 2) return '';
  return `<div class="handoff-addrs">${_addresses.map(a => `
    <button class="handoff-addr ${a.ip === _selected ? 'is-active' : ''}"
            data-handoff="pick" data-ip="${_esc(a.ip)}"
            title="${_esc(a.ip)}">${_esc(_addrLabel(a))}</button>`).join('')}</div>`;
}

// ── 挂载与交互 ──────────────────────────────────────────────────────────────

/**
 * account.js 每次重绘都会把这段 DOM 换掉，所以倒计时不能挂在元素上，
 * 只能挂在模块里的 `_revealUntil` 上，重绘后按剩余时间把计时器接回去。
 */
export function mountHandoff(rerender) {
  clearTimeout(_timer);
  _timer = null;
  if (!_revealUntil) return;
  const remain = _revealUntil - Date.now();
  if (remain <= 0) { _revealUntil = 0; return; }
  _timer = setTimeout(() => { _revealUntil = 0; rerender(); }, remain);
}

/**
 * 处理「手机接入」里的点击。account.js 把它自己的委托转过来，
 * 认得就返回 true。
 */
export function handleHandoffClick(e, rerender) {
  const btn = e.target.closest('[data-handoff]');
  if (!btn) return false;
  const action = btn.dataset.handoff;

  if (action === 'reveal') {
    _revealUntil = Date.now() + REVEAL_MS;
    rerender();
  } else if (action === 'hide') {
    _revealUntil = 0;
    rerender();
  } else if (action === 'pick') {
    _selected = btn.dataset.ip;
    rerender();
  } else if (action === 'copy') {
    const url = _handoffUrl();
    navigator.clipboard?.writeText(url);
    // 这条链接里带着访问码，说清楚再让人往群里贴
    showToast('已复制。链接里带着访问码，别往群里贴。');
  }
  return true;
}

function _esc(str) {
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
