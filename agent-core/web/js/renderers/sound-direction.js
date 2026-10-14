import { TextRenderer } from './text.js';
import { KvLatestRenderer } from './kv-latest.js';

// Preserve the original JSON presentation; ROS/ASR still use the typed contract.
export function directionDisplayData(message) {
  if (message?.state !== 1) return { state: message?.state === 'stale' ? 'stale' : 'no_event' };
  return {
    state: 'fresh', trigger: 'sound_activity',
    timestamp_ms: Math.floor(message.published_us / 1000),
    angle: message.angle_deg,
    unit: 'deg',
    audio_window_start_us: message.window_start_us,
    audio_window_end_us: message.window_end_us,
  };
}

export const SoundDirectionRenderer = {
  name: 'sound-direction',
  canRender: format => format === 'sensor/sound-direction-v1',
  mode: 'log',
  mount(container) {
    this._renderer = Object.assign({}, this.mode === 'kv' ? KvLatestRenderer : TextRenderer);
    this._clearTimer = null;
    this._renderer.mount(container);
  },
  _display(message) {
    const data = directionDisplayData(message);
    // Latest mode retains the last angle and timestamp, while state changes
    // to no_event/stale so it is not presented as a current valid event.
    this._renderer.onData(new TextEncoder().encode(JSON.stringify(data)));
  },
  onData(buffer) {
    try {
      const message = JSON.parse(new TextDecoder().decode(buffer));
      this._display(message);
      if (this._clearTimer) clearTimeout(this._clearTimer);
      this._clearTimer = message?.state === 1 ? setTimeout(() => {
        this._display({ state: 'stale' });
        this._clearTimer = null;
      }, 2000) : null;
    } catch { /* ignore malformed frames */ }
  },
  unmount() {
    if (this._clearTimer) clearTimeout(this._clearTimer);
    this._clearTimer = null;
    this._renderer?.unmount();
    this._renderer = null;
  },
};
