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
