/**
 * monitor-mode.js — Mode toggle controller.
 * Switches between "配置" (configure) and "监控" (monitor) modes.
 */

import { activate, deactivate, resetLayout } from './monitor-dashboard.js';
import { redrawCanvas } from './canvas.js';

let _active = false;

export function initMonitorMode() {
  const toggle = document.getElementById('mode-toggle');
  if (!toggle) return;

  const btns = toggle.querySelectorAll('.mode-toggle-btn');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.mode;
      if (mode === 'monitor' && !_active) {
        _enterMonitor(btns);
      } else if (mode === 'configure' && _active) {
        _exitMonitor(btns);
      }
    });
  });

  // Reset layout button
  const resetBtn = document.getElementById('monitor-layout-reset');
  if (resetBtn) resetBtn.addEventListener('click', resetLayout);
}

function _enterMonitor(btns) {
  _active = true;
  _updateToggleUI(btns, 'monitor');
  document.getElementById('app').classList.add('monitor-active');

  activate();
}

function _exitMonitor(btns) {
  _active = false;
  _updateToggleUI(btns, 'configure');
  document.getElementById('app').classList.remove('monitor-active');

  deactivate();
  // canvas-area was display:none while in monitor mode; getBoundingClientRect()
  // returns zeros for hidden elements, so connection lines are mispositioned.
  // Redraw after the DOM is visible again (next frame ensures layout is flushed).
  requestAnimationFrame(() => redrawCanvas());
}

function _updateToggleUI(btns, activeMode) {
  btns.forEach(b => {
    b.classList.toggle('active', b.dataset.mode === activeMode);
  });
}
