/**
 * local-address.js — 这台机器在局域网上的地址，以及把它填进 URL 模板。
 *
 * **为什么不用 `location.host`。** 机器上本地开的浏览器看到的是 `localhost`，
 * SSH 端口转发看到的是 `127.0.0.1`，反向代理后面看到的是代理的地址 —— 这三种
 * 都是手机拨不通的地址。二维码里只要出现其中一个，扫的人就得到一个打不开的
 * 链接，而且没有任何报错提示他为什么打不开。所以地址问服务端要
 * （`GET /api/network/reachable`，`src/api/network.py`），那边是从网卡上枚举的。
 *
 * 两个使用方：「我的」里的手机接入卡片（handoff.js），和插件配置页里的二维码
 * 字段（qr-field.js）。**地址列表共用，选中项各自持有** —— 在「我的」里选了
 * eth0 不该牵动某个插件的配置页，那是两件无关的事。
 */

let _addresses = null;          // null = 还没问过；[] = 问过了但一个都没有
let _inflight = null;           // 同一帧里两个使用方同时要，只发一个请求

/**
 * 本机可达地址，带缓存。
 *
 * 只问一次：网卡不会在用户看这一页的时候变，而每次重绘都发一个请求会让
 * 配置弹窗每次打开都闪一下。
 */
export async function loadLocalAddresses() {
  if (_addresses) return _addresses;
  if (_inflight) return _inflight;

  _inflight = (async () => {
    let addresses;
    try {
      const json = await (await fetch('/api/network/reachable')).json();
      addresses = json.data?.addresses || [];
    } catch {
      addresses = [];
    }
    // 服务端一个地址都给不出时，退回当前这个 —— 至少在「电脑和手机连同一个
    // 网、用 IP 打开的控制台」这种最常见的情形下是对的。
    if (!addresses.length && location.hostname && !isLoopback(location.hostname)) {
      addresses = [{ device: '', ip: location.hostname, kind: 'other', primary: true }];
    }
    _addresses = addresses;
    _inflight = null;
    return _addresses;
  })();
  return _inflight;
}

/** 已经拿到的地址；还没拉过时是 null（而不是空数组 —— 两者含义不同）。 */
export function cachedLocalAddresses() { return _addresses; }

export function isLoopback(host) {
  return host === 'localhost' || host === '::1' || String(host).startsWith('127.');
}

// ── URL 模板 ────────────────────────────────────────────────────────────────

/**
 * 支持的占位符。刻意只有这四个：模板是驱动作者写的，能被替换的东西越少，
 * 能替换出什么就越容易一眼看清。
 */
const PLACEHOLDERS = ['scheme', 'host', 'port', 'origin'];

/**
 * 把 `{scheme}` `{host}` `{port}` `{origin}` 填进模板。
 *
 * `{origin}` 用的是**选中的网卡地址**拼出来的，不是 `location.origin` —— 见本
 * 文件顶部。这是整个模板机制存在的理由，写错了功能就白做了。
 *
 * 不认识的占位符原样留着：吞掉它只会让驱动作者以为自己写对了。
 */
export function fillTemplate(template, { host = '', port = '', scheme = '' } = {}) {
  if (!template) return '';
  const origin = host ? `${scheme}://${host}${port ? ':' + port : ''}` : '';
  const values = { scheme, host, port, origin };
  return String(template).replace(/\{(\w+)\}/g, (whole, name) => {
    const key = name.toLowerCase();
    return PLACEHOLDERS.includes(key) ? values[key] : whole;
  });
}

/**
 * 模板用没用到本机地址。App Store 这类固定外链不需要网卡选择器，给它画一个
 * 只会让人以为切换网卡能换出不同的 App。
 */
export function referencesHost(template) {
  return /\{(host|origin)\}/i.test(String(template || ''));
}

/**
 * 模板要不要访问 token。
 *
 * 要就拒绝渲染。token 能驱动电机，而 configSchema 是驱动作者写的 —— 不该由
 * 它决定把这把钥匙发给谁。「我的」里那张卡片是本机内置的、带遮挡和自动收起
 * 的，和这个不是一回事。
 */
export function requestsToken(template) {
  return /\{token\}/i.test(String(template || ''));
}
