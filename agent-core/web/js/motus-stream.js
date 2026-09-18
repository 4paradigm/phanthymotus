/**
 * motus-stream.js — WebSocket client for /ws/motus.
 * Emits events to registered listeners.
 *
 * The connection also carries the tab's session_id: the backend treats it as the
 * canvas editor lock's liveness signal and releases the lock shortly after this
 * socket drops. So there must be exactly one connection per tab — connectMotus()
 * is idempotent for that reason (it is called from both app.js and dashboard.js).
 */

import { sessionId } from './session.js';

let _ws = null;
let _retryDelay = 1000;
let _listeners = [];  // Array of { mcpId: string|null, fn: Function }
let _statusCbs = [];  // Status callbacks from every connectMotus() caller
let _lastStatus = 'connecting';

export function onMotusEvent(mcpId, fn) {
  _listeners.push({ mcpId, fn });
}

// Resolves true once the socket is actually open, or false after `timeoutMs`.
// push_event() on the backend is fire-and-forget to whatever clients are
// registered *right now* — an event fired while this tab's socket is still
// reconnecting (page just loaded, brief network blip, backend restart) is
// gone forever, no buffering. Callers that are about to trigger a
// server-pushed event they need to see (e.g. the start-project modal) should
// await this first so they don't silently miss it.
export function whenMotusConnected(timeoutMs = 8000) {
  if (_lastStatus === 'connected') return Promise.resolve(true);
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      _statusCbs = _statusCbs.filter((cb) => cb !== onStatus);
      resolve(false);
    }, timeoutMs);
    function onStatus(state) {
      if (state === 'connected') {
        clearTimeout(timer);
        _statusCbs = _statusCbs.filter((cb) => cb !== onStatus);
        resolve(true);
      }
    }
    _statusCbs.push(onStatus);
  });
}

export function offMotusEvent(fn) {
  _listeners = _listeners.filter(l => l.fn !== fn);
}

function _notify(state) {
  _lastStatus = state;
  _statusCbs.forEach(cb => { try { cb(state); } catch { /* ignore */ } });
}

export function connectMotus(onStatusChange) {
  if (onStatusChange) {
    _statusCbs.push(onStatusChange);
    onStatusChange(_lastStatus);   // late caller still gets the current state
  }
  // Already connected/connecting: reuse it. Opening a second socket would leak
  // the old one (still open, just unreferenced) and make this tab look like two
  // live sessions to the editor lock.
  if (_ws && (_ws.readyState === WebSocket.OPEN || _ws.readyState === WebSocket.CONNECTING)) {
    return;
  }
  function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const token = localStorage.getItem('phanthy_access_token') || '';
    _ws = new WebSocket(`${proto}://${location.host}/ws/motus`
      + `?token=${encodeURIComponent(token)}&session_id=${encodeURIComponent(sessionId())}`);

    _ws.onopen = () => {
      _retryDelay = 1000;
      _notify('connected');
    };

    _ws.onmessage = (e) => {
      let event;
      try { event = JSON.parse(e.data); } catch { return; }
      dispatch(event);
    };

    _ws.onclose = () => {
      _notify('connecting');
      setTimeout(connect, _retryDelay);
      _retryDelay = Math.min(_retryDelay * 2, 30000);
    };

    _ws.onerror = () => {
      _notify('error');
    };
  }
  connect();
}

function dispatch(event) {
  // A pairing request needs to reach the operator even when the Peers panel is
  // closed — requiring the panel to be open would defeat the point of human
  // confirmation. Handled here rather than via a listener because no panel is
  // guaranteed to have subscribed.
  if (event?.type === 'peer_pair_request') {
    import('./peers.js')
      .then((m) => m.onPairRequest(event.payload || {}))
      .catch(() => {});
  }

  _listeners.forEach(({ mcpId, fn }) => {
    if (mcpId === null || mcpId === event.mcp_id) {
      fn(event);
    }
  });
}
