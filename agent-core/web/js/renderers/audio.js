/** audio.js — Rolling waveform renderer for audio/pcm stream (min/max per column) + live playback. */
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
  _PREBUF_CHUNKS: 3,    // 首包预载：攒够 3 个 chunk 再开始播放

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

    this._raf = requestAnimationFrame(() => this._draw());
  },

  onData(buffer, fmt) {
    if (!buffer || buffer.byteLength === 0 || buffer.byteLength % 2 !== 0) return;
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
    if (this._playBtn) {
      this._playBtn.textContent = '▶';
      this._playBtn.title = '播放实时音频';
      this._playBtn.classList.remove('active');
    }
  },

  _feedPlayback(buffer) {
    if (!this._audioCtx || !this._playing) return;

    // Pre-buffering: collect first N chunks before starting playback
    if (this._prebufQueue) {
      this._prebufQueue.push(buffer);
      this._prebufCount++;
      if (this._prebufCount >= this._PREBUF_CHUNKS) {
        // Flush all prebuffered chunks
        const queue = this._prebufQueue;
        this._prebufQueue = null;
        for (const buf of queue) {
          this._scheduleChunk(buf);
        }
      }
      return;
    }

    this._scheduleChunk(buffer);
  },

  _scheduleChunk(buffer) {
    const ctx = this._audioCtx;
    if (!ctx) return;

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
      // Buffer underrun — restart from current time + small delay
      this._nextStartTime = currentTime + 0.05;
    }

    source.start(this._nextStartTime);
    this._nextStartTime += audioBuffer.duration;
  },

  // ── Waveform drawing ──────────────────────────────────────────────────────

  _draw() {
    if (!this._canvas || !this._ctx2d) return;

    const cw = this._canvas.offsetWidth;
    const ch = this._canvas.offsetHeight;
    if (cw > 0 && (this._canvas.width !== cw || this._canvas.height !== ch)) {
      this._canvas.width  = cw * devicePixelRatio;
      this._canvas.height = ch * devicePixelRatio;
      this._canvas.style.width  = cw + 'px';
      this._canvas.style.height = ch + 'px';
      this._ctx2d.scale(devicePixelRatio, devicePixelRatio);
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
