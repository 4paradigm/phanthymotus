/**
 * qr-field.js — 插件配置页里的二维码（`"format": "qr"`）。
 *
 * 有些插件要用户拿手机做点什么：装一个配套 App、用 App 连上这台机器人配网。
 * 这件事只有那个插件知道，所以二维码长在**它自己的配置页**里，由驱动在
 * `configSchema` 里声明：
 *
 *   "install": {
 *     "type": "string", "format": "qr", "description": "扫码安装手机 App",
 *     "x-qr-options": [
 *       {"label": "iOS",     "url": "https://apps.apple.com/app/id123"},
 *       {"label": "Android", "url": "{origin}/downloads/motus.apk"},
 *       {"label": "配网",    "url": "motus://setup?h={host}&p={port}"}
 *     ]
 *   }
 *
 * 三点：
 *
 * 1. **只读，不产生配置值。** 节点上不挂 `data-key`，而保存时两个弹窗都是
 *    `bodyEl.querySelectorAll('[data-key]')`（sidebar.js），所以它天然不进
 *    保存的值里 —— 保存逻辑一行都不用改。
 *
 * 2. **地址来自网卡，不是 `location.host`。** 理由见 local-address.js。
 *
 * 3. **不给 token。** 模板里写 `{token}` 一律拒绝渲染并说明。token 能驱动
 *    电机，而 configSchema 是驱动作者写的 —— 不该由它决定把钥匙发给谁。
 */

import { loadLocalAddresses, cachedLocalAddresses, fillTemplate,
         referencesHost, requestsToken } from './local-address.js';
import { qrSvg } from './qrcode.js';
import { showToast } from './toast.js';

/** 这个字段是纯展示、不收集任何值吗？ */
export function isDisplayOnly(def) {
  return !!def && def.format === 'qr';
}

/**
 * schema 里声明的选项。容错到底：驱动作者写错了应该少一个选项，而不是整个
 * 配置弹窗白屏。
 */
export function qrOptions(def) {
  const raw = def?.['x-qr-options'];
  if (!Array.isArray(raw)) return [];
  return raw
    .filter(o => o && typeof o.url === 'string' && o.url.trim())
    .map((o, i) => ({ label: String(o.label ?? `选项 ${i + 1}`), url: o.url.trim() }));
}

/**
 * 建一个二维码字段。同步返回节点，地址到了再填 —— 和 channel-select 一样的
 * 路子（先放占位，fetch 回来再补），否则整个配置弹窗要等一个网络请求才出来。
 */
export function makeQrField(key, def) {
  const root = document.createElement('div');
  root.className = 'qr-field';

  const options = qrOptions(def);
  if (!options.length) {
    root.innerHTML = `<p class="qr-field-empty">这个插件没有声明可扫的内容（x-qr-options 为空）。</p>`;
    return root;
  }

  const state = { optionIndex: 0, host: '' };

  const tabs = document.createElement('div');
  tabs.className = 'qr-field-tabs';
  tabs.setAttribute('role', 'tablist');
  // 一个选项不画切换条 —— 只有一个选项的切换条是噪音
  if (options.length > 1) root.appendChild(tabs);

  const body = document.createElement('div');
  body.className = 'qr-field-body';
  root.appendChild(body);

  function currentUrl() {
    const tpl = options[state.optionIndex].url;
    return fillTemplate(tpl, {
      host: state.host,
      port: location.port,
      scheme: location.protocol.replace(':', ''),
    });
  }

  function renderTabs() {
    if (options.length < 2) return;
    tabs.innerHTML = '';
    options.forEach((opt, i) => {
      const btn = document.createElement('button');
      btn.type = 'button';           // 弹窗里有 form，默认 submit 会把弹窗提交掉
      btn.className = `qr-field-tab${i === state.optionIndex ? ' is-active' : ''}`;
      btn.textContent = opt.label;
      btn.setAttribute('role', 'tab');
      btn.setAttribute('aria-selected', String(i === state.optionIndex));
      btn.addEventListener('click', () => {
        if (state.optionIndex === i) return;
        state.optionIndex = i;
        renderTabs();
        renderBody();
      });
      tabs.appendChild(btn);
    });
  }

  function renderBody() {
    const tpl = options[state.optionIndex].url;
    body.innerHTML = '';

    // 拒绝带 token 的模板。说出来而不是静默不画 —— 驱动作者要知道为什么没图，
    // 用户要知道这不是坏了。
    if (requestsToken(tpl)) {
      const warn = document.createElement('p');
      warn.className = 'qr-field-refused';
      warn.textContent = '该驱动请求把本机访问码放进二维码，已拒绝。访问码能驱动电机，'
                       + '只有「我的 → 手机接入」才会给出带访问码的二维码。';
      body.appendChild(warn);
      return;
    }

    const url = currentUrl();
    const addresses = cachedLocalAddresses();
    const needsHost = referencesHost(tpl);

    const code = document.createElement('div');
    code.className = 'qr-field-code';
    if (needsHost && !state.host) {
      // 地址还没回来，或者一个都没有
      code.innerHTML = `<span class="qr-field-hint">${
        addresses === null ? '正在读取本机地址…' : '问不到本机的局域网地址'}</span>`;
    } else {
      try {
        code.innerHTML = qrSvg(url, { margin: 4, color: '#1C1917', background: '#FFFFFF' });
      } catch {
        // 内容超出二维码容量。还有「复制链接」这条路，别让整块空着。
        code.innerHTML = `<span class="qr-field-hint">链接太长，画不成二维码</span>`;
      }
    }
    body.appendChild(code);

    const meta = document.createElement('div');
    meta.className = 'qr-field-meta';

    const urlEl = document.createElement('code');
    urlEl.className = 'qr-field-url';
    urlEl.textContent = url;
    urlEl.title = url;
    meta.appendChild(urlEl);

    // 网卡选择器只在模板真的用到本机地址、且确实有多个地址时才画。固定外链
    // 旁边摆一排网卡，会让人以为换网卡能换出不同的 App。
    if (needsHost && addresses && addresses.length > 1) {
      const pills = document.createElement('div');
      pills.className = 'qr-field-addrs';
      for (const a of addresses) {
        const pill = document.createElement('button');
        pill.type = 'button';
        pill.className = `qr-field-addr${a.ip === state.host ? ' is-active' : ''}`;
        pill.textContent = a.device || a.ip;
        pill.title = a.ip;
        pill.addEventListener('click', () => {
          if (state.host === a.ip) return;
          state.host = a.ip;
          renderBody();
        });
        pills.appendChild(pill);
      }
      meta.appendChild(pills);
    }

    const copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'qr-field-copy';
    copy.textContent = '复制链接';
    copy.addEventListener('click', () => {
      navigator.clipboard?.writeText(currentUrl());
      showToast('已复制链接');
    });
    meta.appendChild(copy);

    body.appendChild(meta);
  }

  renderTabs();
  renderBody();

  // 地址回来后补画。只有用到 {host}/{origin} 的模板会因此改变，但重绘一次
  // 最省心，也让网卡 pill 一并出现。
  loadLocalAddresses().then(addresses => {
    if (!state.host) state.host = addresses[0]?.ip || '';
    renderBody();
  });

  return root;
}
