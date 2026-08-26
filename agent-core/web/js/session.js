/**
 * session.js — Per-tab session id, shared by the canvas editor lock and the
 * /ws/motus connection.
 *
 * sessionStorage (not localStorage) so each tab/window gets its own id — otherwise
 * all tabs of one browser would look like the same editor and could edit
 * concurrently, silently clobbering each other's autosave. Surviving F5 in the
 * same tab is intentional: a reload re-claims its own lock instead of locking
 * itself out.
 *
 * Lives in its own module because both canvas.js and motus-stream.js need the id
 * and neither can depend on the other's init order.
 */

const KEY = 'canvas_session_id';

export function sessionId() {
  let id = sessionStorage.getItem(KEY);
  if (!id) {
    id = 'sess-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    sessionStorage.setItem(KEY, id);
  }
  return id;
}
