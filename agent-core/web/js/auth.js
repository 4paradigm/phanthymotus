/**
 * auth.js — Access token management for Dashboard.
 *
 * - Stores token in localStorage
 * - Patches window.fetch to auto-attach Bearer header
 * - Provides wsUrl() for authenticated WebSocket connections
 */

const TOKEN_KEY = 'phanthy_access_token';

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || '';
}

export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

/**
 * 认下 URL 里带来的 token，存好，然后从地址栏里抹掉。
 *
 * 「手机接入」的二维码就靠这一步：在这之前页面只认 localStorage，`?token=`
 * 只对后端 API 有效，扫码打开只会看到一个登录框 —— 用户扫了码却还要手抄一遍
 * 访问码，这个功能等于不存在。
 *
 * 抹掉是必须的：留在地址栏里，它会跟着截图、分享、书签一起走，而它是一把
 * 能驱动电机的钥匙。`replaceState` 同时保证它不进浏览历史。
 *
 * 也接受 `#token=`：有些扫码器会把查询串洗掉，fragment 反而留得住。
 */
export function consumeTokenFromUrl() {
  const url = new URL(location.href);
  let token = url.searchParams.get('token');
  if (!token && url.hash.startsWith('#token=')) {
    token = decodeURIComponent(url.hash.slice('#token='.length));
  }
  if (!token) return false;

  setToken(token);
  url.searchParams.delete('token');
  if (url.hash.startsWith('#token=')) url.hash = '';
  history.replaceState(null, '', url.pathname + url.search + url.hash);
  return true;
}

export async function verifyToken(token) {
  try {
    const res = await _origFetch('/api/auth/verify', {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return res.ok;
  } catch {
    return false;
  }
}

/** Build authenticated WebSocket URL. */
export function wsUrl(path) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const token = getToken();
  const sep = path.includes('?') ? '&' : '?';
  return `${proto}//${location.host}${path}${sep}token=${encodeURIComponent(token)}`;
}

// ── Patch fetch to auto-attach token ─────────────────────────────────────────

const _origFetch = window.fetch.bind(window);

window.fetch = function(url, opts = {}) {
  const token = getToken();
  if (token && typeof url === 'string' && url.startsWith('/api/')) {
    if (!opts.headers) opts.headers = {};
    if (opts.headers instanceof Headers) {
      if (!opts.headers.has('Authorization')) {
        opts.headers.set('Authorization', `Bearer ${token}`);
      }
    } else {
      if (!opts.headers['Authorization']) {
        opts.headers['Authorization'] = `Bearer ${token}`;
      }
    }
  }
  return _origFetch(url, opts);
};
