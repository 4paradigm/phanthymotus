/**
 * mic-stream.js — 浏览器麦克风采集 + 重采样 16kHz + WebSocket 发送
 *
 * 用法：
 *   import { toggleMicStream, isMicActive } from './mic-stream.js';
 *   await toggleMicStream(wsUrl, onStateChange);
 */

let _stream = null;
let _audioCtx = null;
let _ws = null;
let _workletNode = null;
let _active = false;

/**
 * 切换麦克风状态。
 * @param {string} wsUrl  驱动 WebSocket 地址, e.g. ws://localhost:15710/ws/mic
 * @param {(active: boolean) => void} onStateChange  状态回调
 */
export async function toggleMicStream(wsUrl, onStateChange) {
  if (_active) {
    _stopMic();
    onStateChange(false);
    return;
  }

  try {
    _stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true },
    });

    // 目标采样率 16kHz
    const TARGET_RATE = 16000;
    // 浏览器可能不支持 16kHz AudioContext，用原生采样率 + 重采样
    _audioCtx = new AudioContext({ sampleRate: TARGET_RATE });

    // 如果浏览器忽略 sampleRate hint，需要手动重采样
    const actualRate = _audioCtx.sampleRate;
    const downsampleRatio = actualRate / TARGET_RATE;

    const source = _audioCtx.createMediaStreamSource(_stream);

    // 使用 ScriptProcessorNode (广泛兼容)
    // bufferSize 4096 → ~256ms at 16kHz, 足够低延迟
    const bufferSize = 4096;
    const processor = _audioCtx.createScriptProcessor(bufferSize, 1, 1);

    // Connect WebSocket
    _ws = new WebSocket(wsUrl);
    _ws.binaryType = 'arraybuffer';

    await new Promise((resolve, reject) => {
      _ws.onopen = resolve;
      _ws.onerror = reject;
      setTimeout(() => reject(new Error('WS timeout')), 5000);
    });

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
    processor.connect(_audioCtx.destination);

    _workletNode = processor;
    _active = true;
    onStateChange(true);
  } catch (err) {
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
