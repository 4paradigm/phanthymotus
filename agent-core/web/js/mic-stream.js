/**
 * mic-stream.js — 浏览器麦克风采集 + 重采样 16kHz + WebSocket 发送
 *
 * 用法：
 *   import { startMicStream, stopMicStream, isMicActive } from './mic-stream.js';
 *   await startMicStream(wsUrl, onStateChange, async () => ({ deviceId }));
 *   stopMicStream();
 *
 * start 是幂等的，stop 能取消一个还没完成的 start —— 见 startMicStream。
 */

import { deviceConstraint } from './browser-source.js';

let _stream = null;
let _audioCtx = null;
let _ws = null;
let _workletNode = null;
let _active = false;
// 每次 start/stop 加一。一个异步 start 在每个 await 之后核对它，变了就说明中途被
// 停止（或被新的 start 取代），于是释放自己已经拿到的资源并退出。
let _generation = 0;
let _starting = null;

/** The `audio` constraint for getUserMedia. Exported for the tests. */
export function micConstraints(deviceId, rawMic = false) {
  return {
    channelCount: 1, sampleRate: 16000,
    echoCancellation: !rawMic,
    noiseSuppression: !rawMic,
    autoGainControl: !rawMic,
    ...deviceConstraint(deviceId),
  };
}

/**
 * 启动麦克风。已在运行或正在启动时，返回同一个 Promise，不会再开第二路。
 *
 * 两个入口（启动项目、刷新后恢复）会在同一时刻进来。以前的 toggle 要等
 * getUserMedia 和 WebSocket 都完成才置 `_active`，于是两边都看到「未启动」，各开
 * 一路：前一路的音轨和 WS 被覆盖、再也关不掉，/remote_control/mic 上是两路音频。
 *
 * `getOptions` 是在**拿到代次之后**才调用的异步函数（读卡片配置）。放在里面而不是
 * 由调用方先读好，是因为读配置本身也要花时间：在它返回前点停止，必须也能取消。
 *
 * @param {string} wsUrl  e.g. wss://host/ws/mic
 * @param {(active: boolean) => void} onStateChange
 * @param {() => Promise<{deviceId?: string}>} [getOptions]
 * @returns {Promise<boolean>} true 已在采音；false 启动被 stopMicStream 取消
 */
export function startMicStream(wsUrl, onStateChange, getOptions) {
  if (_active) return Promise.resolve(true);
  if (_starting) return _starting;
  const gen = ++_generation;
  const run = _start(gen, wsUrl, onStateChange, getOptions);
  _starting = run;
  const clear = () => { if (_starting === run) _starting = null; };
  run.then(clear, clear);
  return run;
}

/** 停止麦克风，包括取消一个还在进行中的 start。空闲时什么也不做。 */
export function stopMicStream() {
  _generation++;
  _starting = null;
  _stopMic();
}

/**
 * 切换麦克风状态。保留给旧调用方；新代码用 start/stop。
 * @param {string} wsUrl  驱动 WebSocket 地址, e.g. ws://localhost:15710/ws/mic
 * @param {(active: boolean) => void} onStateChange  状态回调
 * @param {() => Promise<{deviceId?: string}>} [getOptions]
 */
export async function toggleMicStream(wsUrl, onStateChange, getOptions) {
  if (_active || _starting) {
    stopMicStream();
    onStateChange(false);
    return;
  }
  await startMicStream(wsUrl, onStateChange, getOptions);
}

async function _start(gen, wsUrl, onStateChange, getOptions) {
  // 先放在局部变量里，全部就绪才提交到模块状态。中途被取消时，stopMicStream
  // 已经清过模块状态，只有这里的局部资源还需要释放。
  let stream = null, audioCtx = null, ws = null;
  const cancelled = () => gen !== _generation;
  const release = () => {
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (audioCtx) audioCtx.close().catch(() => {});
    if (ws) { try { ws.close(); } catch { /* already closed */ } }
  };

  try {
    const opts = (getOptions ? await getOptions() : null) || {};
    if (cancelled()) return false;
    const deviceId = opts.deviceId || '';

    // The browser's own echo canceller, noise suppressor and AGC, on by
    // default because this microphone is usually an operator's laptop with the
    // agent's audio coming out of the same speakers — exactly what they are for.
    //
    // They also make this path useless as a test rig for the robot's own echo
    // cancellation: the browser plays the TTS itself, so its canceller has a
    // perfect reference and removes the echo before a single sample reaches
    // Agent Core. The perception card downstream then has nothing to cancel,
    // with or without it, which is indistinguishable from the card not working.
    //
    // `localStorage.phanthy_raw_mic = '1'` (or ?rawmic=1) turns all three off
    // and hands over whatever the microphone actually heard. AGC matters as
    // much as the canceller here: it changes the gain over time, which makes
    // the echo path time-varying and is a different thing to debug.
    const rawMic = localStorage.getItem('phanthy_raw_mic') === '1'
      || new URLSearchParams(location.search).get('rawmic') === '1';
    if (rawMic) console.warn('[mic] raw capture: browser AEC/NS/AGC are OFF');
    stream = await navigator.mediaDevices.getUserMedia({
      audio: micConstraints(deviceId, rawMic),
    });
    if (cancelled()) { release(); return false; }
    // 实际打开的是哪一个。「选了 B、听到的是 A」只有这一行能直接看出来。
    const track = stream.getAudioTracks()[0];
    console.info(`[mic] capturing: ${track ? track.label || '(unlabelled)' : '(no track)'}`
      + (deviceId ? ' (selected device)' : ' (browser default)'));

    // 目标采样率 16kHz
    const TARGET_RATE = 16000;
    // 浏览器可能不支持 16kHz AudioContext，用原生采样率 + 重采样
    audioCtx = new AudioContext({ sampleRate: TARGET_RATE });

    // 如果浏览器忽略 sampleRate hint，需要手动重采样
    const actualRate = audioCtx.sampleRate;
    const downsampleRatio = actualRate / TARGET_RATE;

    const source = audioCtx.createMediaStreamSource(stream);

    // 使用 ScriptProcessorNode (广泛兼容)
    // bufferSize 4096 → ~256ms at 16kHz, 足够低延迟
    const bufferSize = 4096;
    const processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);

    // Connect WebSocket
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    let wsTimer;
    await new Promise((resolve, reject) => {
      ws.onopen = resolve;
      ws.onerror = () => reject(new Error('麦克风 WebSocket 连接失败'));
      wsTimer = setTimeout(() => reject(new Error('麦克风 WebSocket 连接超时')), 5000);
    }).finally(() => clearTimeout(wsTimer));
    if (cancelled()) { release(); return false; }

    // 提交。此后由 _stopMic 负责释放。
    _stream = stream; _audioCtx = audioCtx; _ws = ws;
    stream = audioCtx = ws = null;

    processor.onaudioprocess = (e) => {
      if (!_ws || _ws.readyState !== WebSocket.OPEN) return;
      const float32 = e.inputBuffer.getChannelData(0);

      let samples;
      if (Math.abs(downsampleRatio - 1) < 0.01) {
        samples = float32;
      } else {
        // 简单线性重采样
        const outLen = Math.floor(float32.length / downsampleRatio);
        samples = new Float32Array(outLen);
        for (let i = 0; i < outLen; i++) {
          samples[i] = float32[Math.floor(i * downsampleRatio)];
        }
      }

      // Float32 → Int16 PCM
      const int16 = new Int16Array(samples.length);
      for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

      _ws.send(int16.buffer);
    };

    source.connect(processor);
    // Connect to a silent destination (processor must be connected to stay alive, but we don't want playback)
    const silentGain = _audioCtx.createGain();
    silentGain.gain.value = 0;
    processor.connect(silentGain);
    silentGain.connect(_audioCtx.destination);

    _workletNode = processor;
    _active = true;
    onStateChange(true);
    return true;
  } catch (err) {
    release();
    // 被取消的 start 失败了也不算失败：停止的人已经得到了想要的结果，而且这时
    // 的 _stopMic 会误伤一个更新的 start。
    if (cancelled()) return false;
    console.error('[mic-stream] Failed to start:', err);
    _stopMic();
    onStateChange(false);
    throw err;
  }
}

function _stopMic() {
  if (_stream) {
    _stream.getTracks().forEach(t => t.stop());
    _stream = null;
  }
  if (_workletNode) {
    _workletNode.disconnect();
    _workletNode = null;
  }
  if (_audioCtx) {
    _audioCtx.close().catch(() => {});
    _audioCtx = null;
  }
  if (_ws) {
    _ws.close();
    _ws = null;
  }
  _active = false;
}

export function isMicActive() {
  return _active;
}
