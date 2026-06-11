/**
 * monitor-mode.js — Mode toggle controller.
 * Switches between "配置" (configure) and "监控" (monitor) modes.
 */

import { activate, deactivate, resetLayout } from './monitor-dashboard.js';

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
}

function _updateToggleUI(btns, activeMode) {
  btns.forEach(b => {
    b.classList.toggle('active', b.dataset.mode === activeMode);
  });
}
