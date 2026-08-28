/** audio.js — Rolling waveform renderer for audio/pcm stream (min/max per column) + live playback. */

/**
 * End-of-utterance marker on the TTS audio protocol, published by both TTS
 * engines after every utterance and consumed by the speaker drivers
 * (`unitree/g1/device.py`, `unitree/r1/device.py`) as `len(pcm) == 8 and pcm ==
 * AUDIO_EOF_MAGIC`. It is protocol, not audio: fed to the waveform as samples it
 * reported `音频流 0ms/帧` after every announcement — 8 bytes is 4 samples, which
 * rounds to 0 ms — which reads as a broken stream.
 */
const AUDIO_EOF = [0x01, 0x00, 0xff, 0xff, 0x01, 0x00, 0xff, 0xff];

export function isAudioEof(buffer) {
  if (!buffer || buffer.byteLength !== AUDIO_EOF.length) return false;
  const b = new Uint8Array(buffer);
  return AUDIO_EOF.every((v, i) => b[i] === v);
}

export const AudioRenderer = {
  name: 'audio',
  canRender: (hint) => hint && hint.startsWith('audio/'),

  _el:       null,
  _canvas:   null,
  _ctx2d:    null,
  _ring:     null,
  _ringLen:  16000,  // 1 second @ 16kHz — ring buffer of raw samples
  _writePos: 0,
  _raf:      null,
  _label:    null,

  // Playback state
  _audioCtx:      null,
  _playing:       false,
  _playBtn:       null,
  _nextStartTime: 0,
  _prebufCount:   0,
  _prebufQueue:   null,
  // Scheduled buffers, so ⏸ can actually stop them. Created per instance in
  // mount(): renderers are cloned with Object.assign, so a Set built here on
  // the prototype would be shared and pausing one card would cut another's audio.
  _sources:       null,
  _PREBUF_CHUNKS: 5,     // 首包预载：攒够 5 个 chunk (~500ms) 再开始播放
  _UNDERRUN_LEAD: 0.20,  // 欠载重启时给出的余量，秒
  _MAX_LEAD:      1.5,   // 允许领先播放头的上限，超过则丢帧（绝不回拨时间轴）
  _dropped:       0,     // 因超前而丢弃的帧数
  _drawnPos:      -1,

  mount(container) {
    this._el = document.createElement('div');
    this._el.className = 'renderer-audio';

    this._canvas = document.createElement('canvas');
    this._canvas.className = 'audio-waveform';
    this._el.appendChild(this._canvas);

    // Bottom bar: label + play button
    const bar = document.createElement('div');
    bar.className = 'audio-bar';

    this._label = document.createElement('div');
    this._label.className = 'audio-label';
    this._label.textContent = '等待音频流…';
    bar.appendChild(this._label);

    this._playBtn = document.createElement('button');
    this._playBtn.className = 'audio-play-btn';
    this._playBtn.textContent = '▶';
    this._playBtn.title = '播放实时音频';
    this._playBtn.addEventListener('click', () => this._togglePlay());
    bar.appendChild(this._playBtn);

    this._el.appendChild(bar);
    container.appendChild(this._el);

    this._ctx2d = this._canvas.getContext('2d');
    this._ring = new Float32Array(this._ringLen);
    this._writePos = 0;
    this._drawnPos = -1;
    this._sources = new Set();
    this._dropped = 0;

    this._raf = requestAnimationFrame(() => this._draw());
  },

  onData(buffer, fmt) {
    if (!buffer || buffer.byteLength === 0 || buffer.byteLength % 2 !== 0) return;
    if (isAudioEof(buffer)) {
      // Protocol frame, not samples. Keep the tail of the waveform on screen and
      // say what happened instead of reporting a 0 ms frame.
      if (this._label) this._label.textContent = '○ 音频流  本句结束';
      // An utterance boundary is exactly when the jitter buffer has to be
      // re-armed. The prebuffer used to be spent on the first utterance and
      // never rebuilt, so every later one started from _scheduleChunk's
      // underrun branch — a fixed, tiny lead — and gapped on the first hiccup.
      // Flush first: an utterance shorter than _PREBUF_CHUNKS never fills the
      // prebuffer, and re-arming without flushing would discard it unplayed.
      // _nextStartTime is deliberately left alone: it is the end of the audio
      // already handed to the audio thread, and resetting it would schedule the
      // next utterance on top of this one's still-playing tail.
      if (this._playing) {
        this._flushPrebuf();
        this._prebufQueue = [];
        this._prebufCount = 0;
      }
      return;
    }
    const pcm = new Int16Array(buffer);
    const ring = this._ring;
    const len = this._ringLen;
    for (let i = 0; i < pcm.length; i++) {
      ring[this._writePos % len] = pcm[i] / 32768;
      this._writePos++;
    }
    if (this._label) {
      this._label.textContent = `● 音频流  ${Math.round(pcm.length / 16)}ms/帧`;
    }
    // Feed playback
    if (this._playing) {
      this._feedPlayback(buffer);
    }
  },

  onDataSilent(buffer) {
    if (!buffer || buffer.byteLength === 0 || buffer.byteLength % 2 !== 0) return;
    if (isAudioEof(buffer)) return;
    const pcm = new Int16Array(buffer);
    const ring = this._ring;
    const len = this._ringLen;
    for (let i = 0; i < pcm.length; i++) {
      ring[this._writePos % len] = pcm[i] / 32768;
      this._writePos++;
    }
  },

  clear() {
    if (this._ring) this._ring.fill(0);
    this._writePos = 0;
    this._drawnPos = -1;
    if (this._label) this._label.textContent = '等待音频流…';
  },

  stopPlayback() {
    this._stopPlay();
    this.clear();
  },

  // ── Playback control ──────────────────────────────────────────────────────

  _togglePlay() {
    if (this._playing) {
      this._stopPlay();
    } else {
      this._startPlay();
    }
  },

  _startPlay() {
    if (!this._audioCtx) {
      this._audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    }
    if (this._audioCtx.state === 'suspended') {
      this._audioCtx.resume();
    }

    this._playing = true;
    this._nextStartTime = 0;
    this._prebufCount = 0;
    this._prebufQueue = [];

    if (this._playBtn) {
      this._playBtn.textContent = '⏸';
      this._playBtn.title = '暂停播放';
      this._playBtn.classList.add('active');
    }
  },

  _stopPlay() {
    this._playing = false;
    this._prebufQueue = null;
    this._prebufCount = 0;
    // Stop the buffers already handed to the audio thread. Setting _playing
    // false only stops *scheduling* new ones; everything queued ahead of the
    // playhead kept sounding, so ⏸ appeared to do nothing and only closing the
    // panel (which calls unmount → audioCtx.close) actually silenced it.
    // Raising _MAX_LEAD to 1.5s made a long-standing bug obvious: with the old
    // ~50ms of lead there was barely anything queued to keep playing.
    for (const source of this._sources || []) {
      try { source.onended = null; source.stop(); } catch { /* already ended */ }
    }
    this._sources?.clear();
    this._nextStartTime = 0;
    if (this._playBtn) {
      this._playBtn.textContent = '▶';
      this._playBtn.title = '播放实时音频';
      this._playBtn.classList.remove('active');
    }
  },

  _flushPrebuf() {
    if (!this._prebufQueue || this._prebufQueue.length === 0) return;
    const queue = this._prebufQueue;
    this._prebufQueue = null;
    this._prebufCount = 0;
    for (const buf of queue) {
      this._scheduleChunk(buf);
    }
  },

  _feedPlayback(buffer) {
    if (!this._audioCtx || !this._playing) return;

    // Pre-buffering: collect first N chunks before starting playback
    if (this._prebufQueue) {
      this._prebufQueue.push(buffer);
      this._prebufCount++;
      if (this._prebufCount >= this._PREBUF_CHUNKS) {
        this._flushPrebuf();
      }
      return;
    }

    this._scheduleChunk(buffer);
  },

  _scheduleChunk(buffer) {
    const ctx = this._audioCtx;
    if (!ctx) return;
    if (ctx.state === 'suspended') {
      ctx.resume().then(() => this._scheduleChunk(buffer));
      return;
    }

    const pcm = new Int16Array(buffer);
    const numSamples = pcm.length;
    const audioBuffer = ctx.createBuffer(1, numSamples, 16000);
    const channelData = audioBuffer.getChannelData(0);

    // Convert Int16 to Float32 — no fade needed, chunks are continuous PCM
    for (let i = 0; i < numSamples; i++) {
      channelData[i] = pcm[i] / 32768;
    }

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);

    // Schedule playback time
    const currentTime = ctx.currentTime;
    if (this._nextStartTime < currentTime) {
      // Underrun — every scheduled buffer has already finished, so this only
      // ever moves the schedule *forward*. The old +0.05 here left a 50ms lead
      // as the permanent steady-state margin, so the next delay over 50ms
      // gapped again, and again. _UNDERRUN_LEAD gives the recovery something to
      // work with.
      this._nextStartTime = currentTime + this._UNDERRUN_LEAD;
    } else if (this._nextStartTime - currentTime > this._MAX_LEAD) {
      // Too far ahead. Drop this chunk rather than reschedule: _nextStartTime is
      // the end of audio already handed to the audio thread, so assigning
      // `currentTime + _MAX_LEAD` here — which is what this used to do — placed
      // the next buffer *inside* the previous one. That is what made playback
      // overlap and run fast: once the lead was pinned at the cap, every frame
      // was rewound onto the one before it, so 100ms of audio played every 70ms.
      // Advancing only by `+= duration` makes contiguity structural.
      this._dropped = (this._dropped || 0) + 1;
      if (this._label) {
        this._label.textContent = `● 音频流  丢帧 ${this._dropped}（缓冲超前）`;
      }
      return;
    }

    source.start(this._nextStartTime);
    this._nextStartTime += audioBuffer.duration;
    // Track it so _stopPlay can actually silence what is already queued.
    // Self-removing on end, so the set only ever holds pending buffers.
    this._sources?.add(source);
    source.onended = () => this._sources?.delete(source);
  },

  // ── Waveform drawing ──────────────────────────────────────────────────────

  _draw() {
    if (!this._canvas || !this._ctx2d) return;

    const cw = this._canvas.offsetWidth;
    const ch = this._canvas.offsetHeight;

    // The waveform shares the main thread with _scheduleChunk, which has only a
    // few hundred ms of scheduling lead to work with. A full redraw is a
    // per-column fillRect plus a scan of the whole ring, so skip the ones that
    // cannot change anything the user sees: a hidden tab, and frames where
    // neither new samples nor a resize arrived (audio is 10 fps, this is 60).
    const wantW = Math.round(cw * devicePixelRatio);
    const wantH = Math.round(ch * devicePixelRatio);
    const resized = this._canvas.width !== wantW || this._canvas.height !== wantH;
    if (document.hidden || (!resized && this._writePos === this._drawnPos)) {
      this._raf = requestAnimationFrame(() => this._draw());
      return;
    }
    this._drawnPos = this._writePos;

    // Assign the backing-store size, not the CSS size: after a resize
    // canvas.width is cw * devicePixelRatio, so `canvas.width !== cw` is always
    // true on a HiDPI display and this reallocated the canvas and reset the
    // transform on every animation frame.
    if (cw > 0 && resized) {
      this._canvas.width  = wantW;
      this._canvas.height = wantH;
      this._canvas.style.width  = cw + 'px';
      this._canvas.style.height = ch + 'px';
      // Setting width/height resets the transform, so re-apply the DPR scale.
      this._ctx2d.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    }
    if (!this._canvas.width) {
      this._raf = requestAnimationFrame(() => this._draw());
      return;
    }

    const w = cw || (this._canvas.width / devicePixelRatio);
    const h = ch || (this._canvas.height / devicePixelRatio);
    const ctx = this._ctx2d;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#1C1C1E';
    ctx.fillRect(0, 0, w, h);

    // Center line
    const mid = h / 2;
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(w, mid);
    ctx.stroke();

    const ring = this._ring;
    const ringLen = this._ringLen;
    const filled = Math.min(this._writePos, ringLen);
    if (filled === 0) {
      this._raf = requestAnimationFrame(() => this._draw());
      return;
    }

    // Compute min/max per pixel column (peak visualization)
    const cols = Math.floor(w);
    const samplesPerCol = filled / cols;
    const amp = mid * 0.9;

    // Determine the start index in ring buffer (oldest sample of the visible window)
    const startIdx = this._writePos >= ringLen
      ? this._writePos % ringLen   // ring is full, start from write position (oldest)
      : 0;                         // ring not full, start from 0

    ctx.fillStyle = '#D97757';

    for (let col = 0; col < cols; col++) {
      const sampleStart = Math.floor(col * samplesPerCol);
      const sampleEnd   = Math.floor((col + 1) * samplesPerCol);
      // A column with no samples of its own must draw nothing. Falling through
      // with the sentinels below left mx at -1, which reads as "full negative
      // amplitude" and painted a 1px bar at mid + amp — a solid line across the
      // bottom of the panel whenever the buffer held fewer samples than the
      // canvas is wide, which is every frame while the stream is silent.
      if (sampleEnd <= sampleStart) continue;
      let mn = 1, mx = -1;
      for (let s = sampleStart; s < sampleEnd; s++) {
        const idx = (startIdx + s) % ringLen;
        const v = ring[idx];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      const y1 = mid - mx * amp;
      const y2 = mid - mn * amp;
      const barH = Math.max(y2 - y1, 1);
      ctx.fillRect(col, y1, 1, barH);
    }

    this._raf = requestAnimationFrame(() => this._draw());
  },

  onEvent(event) {
    if (!this._label) return;
    if (event.type === 'mcp_result') {
      const text = event.payload?.result?.text;
      if (text) this._label.textContent = String(text).slice(0, 120);
    }
  },

  unmount() {
    if (this._raf) { cancelAnimationFrame(this._raf); this._raf = null; }
    this._stopPlay();
    if (this._audioCtx) { this._audioCtx.close(); this._audioCtx = null; }
    this._el?.remove();
    this._el = null; this._canvas = null; this._ctx2d = null;
    this._label = null; this._ring = null;
  },
};
