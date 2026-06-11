/**
 * motus-stream.js — WebSocket client for /ws/motus.
 * Emits events to registered listeners.
 */

let _ws = null;
let _retryDelay = 1000;
let _listeners = [];  // Array of { mcpId: string|null, fn: Function }

export function onMotusEvent(mcpId, fn) {
  _listeners.push({ mcpId, fn });
}

export function offMotusEvent(fn) {
  _listeners = _listeners.filter(l => l.fn !== fn);
}

export function connectMotus(onStatusChange) {
  const _cb = onStatusChange || (() => {});
  function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    _ws = new WebSocket(`${proto}://${location.host}/ws/motus`);

    _ws.onopen = () => {
      _retryDelay = 1000;
      _cb('connected');
    };

    _ws.onmessage = (e) => {
      let event;
      try { event = JSON.parse(e.data); } catch { return; }
      dispatch(event);
    };

    _ws.onclose = () => {
      _cb('connecting');
      setTimeout(connect, _retryDelay);
      _retryDelay = Math.min(_retryDelay * 2, 30000);
    };

    _ws.onerror = () => {
      _cb('error');
    };
  }
  connect();
}

function dispatch(event) {
  _listeners.forEach(({ mcpId, fn }) => {
    if (mcpId === null || mcpId === event.mcp_id) {
      fn(event);
    }
  });
}
