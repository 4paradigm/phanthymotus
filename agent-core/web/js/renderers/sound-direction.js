import { TextRenderer } from './text.js';
import { KvLatestRenderer } from './kv-latest.js';

// Preserve the original JSON presentation; ROS/ASR still use the typed contract.
export function directionDisplayData(message) {
  if (message?.state !== 1) return { state: 'no_event' };
  return {
    state: 'fresh', trigger: 'sound_activity',
    timestamp_ms: Math.floor(message.published_us / 1000),
    angle: ((message.angle_deg % 360) + 360) % 360,
    unit: 'deg', reference: 'robot_front_clockwise',
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
    // The generic latest-value renderer retains omitted keys. Clear them when
    // speech ends so a previous angle cannot look like a current observation.
    if (this.mode === 'kv' && data.state === 'no_event') {
      for (const key of Object.keys(this._renderer._cells)) {
        if (key !== 'state') data[key] = null;
      }
    }
    this._renderer.onData(new TextEncoder().encode(JSON.stringify(data)));
  },
  onData(buffer) {
    try {
      const message = JSON.parse(new TextDecoder().decode(buffer));
      this._display(message);
      if (this._clearTimer) clearTimeout(this._clearTimer);
      this._clearTimer = message?.state === 1 ? setTimeout(() => {
        this._display({ state: 0 });
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
