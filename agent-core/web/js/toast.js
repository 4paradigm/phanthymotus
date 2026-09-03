/**
 * toast.js — transient bottom-centre notice, shared across panels.
 *
 * Extracted from canvas.js, which mounted its toast inside the canvas element.
 * The peer panel needs the same notice but has no canvas to hang it on, so the
 * host is now a parameter and defaults to document.body.
 *
 * Behaviour is deliberately identical to the canvas original (same id, styling,
 * 2.5s animation, 2600ms removal) so that extracting it changes nothing on the
 * canvas — the toast there is load-bearing for edit-permission feedback.
 */

export function showToast(msg, host) {
  const parent = host || document.body;
  const old = document.getElementById('canvas-toast');
  if (old) old.remove();

  const toast = document.createElement('div');
  toast.id = 'canvas-toast';
  toast.textContent = msg;
  // `position:absolute` needs a positioned ancestor. The canvas element is
  // positioned; document.body generally is not, so fall back to fixed when
  // mounting on the body — otherwise the toast scrolls away with the page.
  const position = parent === document.body ? 'fixed' : 'absolute';
  toast.style.cssText =
    `position:${position};bottom:80px;left:50%;transform:translateX(-50%);` +
    'width:fit-content;max-width:80%;background:rgba(28,25,23,.85);color:#fff;' +
    'padding:10px 20px;border-radius:20px;font-size:13px;z-index:9999;' +
    'pointer-events:none;opacity:0;animation:canvas-toast-in 2.5s ease forwards;';
  parent.appendChild(toast);
  setTimeout(() => toast.remove(), 2600);
}
