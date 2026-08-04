#!/usr/bin/env python3
"""
plugins/tts.py — TTSPlugin: sherpa-onnx VITS TTS.

On-device text-to-speech using sherpa-onnx MeloTTS (Chinese + English).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_BYTES = 3200  # 100ms @ 16kHz 16-bit mono

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=200,
    durability=DurabilityPolicy.VOLATILE,
)

TOOLS = [
    {
        "name": "tts",
        "type": "processor",
        "multiInstance": True,
        "description": "TTS — start/stop speech synthesis, speak text, or get status",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "speak", "info", "config"],
                    "description": "Action to perform"
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 topic for text input (data/json, required for action=start)"
                },
                "text": {
                    "type": "string",
                    "description": "Text to synthesize (required for action=speak)"
                },
            },
            "required": ["action"]
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "speaker_id": {"type": "integer", "description": "Speaker ID", "default": 0, "scope": "shared"},
                "speed":      {"type": "number", "description": "Speech speed (1.0 = normal)", "default": 1.0, "scope": "shared"},
            },
            "required": []
        },
        "topic_in":  [{"format": "data/json",     "desc": "text to synthesize"}],
        "topic_out": [{"format": "audio/pcm-16k", "desc": "synthesized PCM audio"}],
    }
]


# ── TTS Adapter ──────────────────────────────────────────────────────────────

class TTSAdapter(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes: ...

    def synthesize_stream(self, text: str):
        """Yield raw PCM bytes as they arrive. Default: collect all."""
        yield self.synthesize(text)


class SherpaOnnxTTSAdapter(TTSAdapter):
    """On-device TTS using sherpa-onnx Matcha (flow-matching, fast non-autoregressive)."""

    def __init__(self, model_dir: str, speaker_id: int = 0, speed: float = 1.0):
        import os
        from utils.model_downloader import ensure_model
        ensure_model("tts", model_dir)
        ensure_model("tts_vocoder", model_dir)

        import sherpa_onnx
        # Matcha model files
        acoustic_model = os.path.join(model_dir, "model-steps-3.onnx")
        vocoder = os.path.join(model_dir, "vocos-16khz-univ.onnx")
        lexicon_path = os.path.join(model_dir, "lexicon.txt")
        tokens_path = os.path.join(model_dir, "tokens.txt")
        data_dir = os.path.join(model_dir, "espeak-ng-data")
        if not os.path.isdir(data_dir):
            data_dir = ""

        # Gather rule FSTs
        rule_fsts = []
        for name in ("date-zh.fst", "number-zh.fst", "phone-zh.fst"):
            p = os.path.join(model_dir, name)
            if os.path.exists(p):
                rule_fsts.append(p)

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                    acoustic_model=acoustic_model,
                    vocoder=vocoder,
                    lexicon=lexicon_path if os.path.exists(lexicon_path) else "",
                    tokens=tokens_path,
                    data_dir=data_dir,
                    length_scale=1.0 / speed if speed else 1.0,
                ),
                num_threads=2,
                provider="cpu",
            ),
            rule_fsts=",".join(rule_fsts) if rule_fsts else "",
        )
        self._tts = sherpa_onnx.OfflineTts(tts_config)
        self._sid = speaker_id
        self._speed = speed
        log.info(f"[tts] sherpa-onnx Matcha loaded: model_dir={model_dir}, "
                 f"speaker_id={speaker_id}, speed={speed}")

    def synthesize(self, text: str) -> bytes:
        return b''.join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        import struct
        audio = self._tts.generate(text, sid=self._sid, speed=self._speed)
        float_samples = audio.samples
        # Matcha + vocos-16khz outputs 16kHz directly, no resampling needed
        pcm = struct.pack(f'<{len(float_samples)}h',
                         *[int(max(-32768, min(32767, s * 32767))) for s in float_samples])
        for i in range(0, len(pcm), CHUNK_BYTES):
            yield pcm[i:i + CHUNK_BYTES]




# ── TRT TTS Adapter (VITS2-Mix, no PyTorch dependency) ────────────────────

_ORT_SESSIONS = {}
_TRT_ENGINES = {}

def _get_ort_session(path):
    if path not in _ORT_SESSIONS:
        import onnxruntime as ort
        _ORT_SESSIONS[path] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _ORT_SESSIONS[path]

def _get_trt_engine(path):
    if path not in _TRT_ENGINES:
        import tensorrt as trt
        with open(path, "rb") as f:
            _TRT_ENGINES[path] = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
    return _TRT_ENGINES[path]


class TRTTSAdapter(TTSAdapter):
    """VITS2-Mix TensorRT TTS adapter. No PyTorch dependency.

    Uses ONNX Runtime for encoder, TRT engines for flow + decoder, NumPy for iSTFT.
    """

    _MODEL_CFG = {
        "v8":   {"n_fft": 64,  "hop": 4, "gain": 0.166},
        "i128": {"n_fft": 128, "hop": 4, "gain": 0.0833},
    }

    MAX_TRT_FRAMES = 3000

    def __init__(self, model_dir: str, trt_dir: str,
                 speaker_id: int = 0, speed: float = 1.0,
                 model_type: str = "v8"):
        self._speed = speed
        self._trt_dir = trt_dir

        sys.path.insert(0, model_dir)
        from frontend.cleaner import clean_text_mix
        from frontend import cleaned_text_to_sequence_mix
        self._clean_text = clean_text_mix
        self._seq_mix = cleaned_text_to_sequence_mix

        mc = self._MODEL_CFG.get(model_type, self._MODEL_CFG["v8"])
        self._n_fft = mc["n_fft"]
        self._hop = mc["hop"]
        self._gain = mc["gain"]

        # Periodic Hann window (matches TorchSTFT fftbins=True)
        w = np.hanning(self._n_fft + 1)[:self._n_fft].astype(np.float32)
        self._window = w.reshape(1, self._n_fft, 1)

        encoder_path = os.path.join(trt_dir, "encoder_duration.onnx")
        self._encoder = _get_ort_session(encoder_path)

        flow_path = os.path.join(trt_dir, "flow.trt")
        dec_path = os.path.join(trt_dir, "decoder.trt")
        self._flow_eng = _get_trt_engine(flow_path)
        self._dec_eng = _get_trt_engine(dec_path)
        if self._flow_eng is None or self._dec_eng is None:
            raise RuntimeError(f"Failed to load TRT engines from {trt_dir}")

        import ctypes
        self._cuda = ctypes.CDLL("libcudart.so")
        self._cuda.cudaMalloc.restype = int
        self._cuda.cudaMalloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._cuda.cudaFree.restype = int
        self._cuda.cudaFree.argtypes = [ctypes.c_void_p]
        self._cuda.cudaMemcpy.restype = int
        self._cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

        log.info("[tts] TRT adapter loaded: model=%s n_fft=%d hop=%d gain=%.4f",
                 model_type, self._n_fft, self._hop, self._gain)

    def _gpu_alloc(self, size):
        ptr = ctypes.c_void_p(0)
        self._cuda.cudaMalloc(ctypes.byref(ptr), size)
        return ptr

    def _trt_run(self, engine, inputs, output_names):
        import tensorrt as trt
        ctx = engine.create_execution_context()
        gpu_ptrs, outputs = {}, {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                data = inputs[name].astype(np.float32)
                ctx.set_input_shape(name, data.shape)
                ptr = self._gpu_alloc(data.nbytes)
                self._cuda.cudaMemcpy(ptr, data.ctypes.data, data.nbytes, 1)
                gpu_ptrs[name] = ptr
            else:
                shape = tuple(ctx.get_tensor_shape(name))
                outputs[name] = np.empty(shape, dtype=np.float32)
                ptr = self._gpu_alloc(outputs[name].nbytes)
                gpu_ptrs[name] = ptr
        bindings = [gpu_ptrs[engine.get_tensor_name(i)].value for i in range(engine.num_io_tensors)]
        ctx.execute_v2(bindings)
        for name, arr in outputs.items():
            self._cuda.cudaMemcpy(arr.ctypes.data, gpu_ptrs[name], arr.nbytes, 2)
        for p in gpu_ptrs.values():
            self._cuda.cudaFree(p)
        return tuple(outputs[n] for n in output_names)

    @staticmethod
    def _split_sentences(text: str) -> list:
        import re
        parts = re.split(r'(?<=[。！？\n])(?![。！？\n])|(?<=[.!?\n])(?![.!?\n])', text)
        chunks, buf = [], ""
        for part in parts:
            if len(buf) + len(part) < 10:
                buf += part
            else:
                if buf.strip(): chunks.append(buf.strip())
                buf = part
        if buf.strip(): chunks.append(buf.strip())
        return chunks if len(chunks) > 1 else [text]

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def _synthesize_chunk(self, text: str, noise_scale: float = 0.667):
        t0 = time.perf_counter()
        norm_text, phones, tones, langs, _ = self._clean_text(text)
        pids, tids, lids = self._seq_mix(phones, tones, langs)
        pids = [0] + [p for pid in pids for p in (pid, 0)]
        tids = [0] + [t for tid in tids for t in (tid, 0)]
        lids = [0] + [l for lid in lids for l in (lid, 0)]
        T = len(pids)

        ph = np.array([pids], dtype=np.int32); to = np.array([tids], dtype=np.int32)
        la = np.array([lids], dtype=np.int32); xl = np.array([T], dtype=np.int32)
        m_p, logs_p, logw, x_mask = self._encoder.run(None, {"ph": ph, "to": to, "la": la, "xl": xl})

        # MAS duration expansion (pure NumPy)
        w = np.exp(logw[0, 0, :T]) * x_mask[0, 0, :T]
        w_ceil = np.ceil(w).astype(np.int32)
        Ty = max(1, int(w_ceil.sum()))
        y_mask = np.ones((1, 1, Ty), dtype=np.float32)
        cum_dur = np.cumsum(w_ceil)
        attn = np.zeros((1, T, Ty), dtype=np.float32)
        for tx in range(T):
            end_pos = int(cum_dur[tx])
            start_pos = end_pos - int(w_ceil[tx])
            if start_pos < Ty and end_pos > 0:
                lo, hi = max(start_pos, 0), min(end_pos, Ty)
                if hi > lo: attn[0, tx, lo:hi] = 1.0
        m_p_exp = np.matmul(m_p[:, :, :T], attn)
        logs_p_exp = np.matmul(logs_p[:, :, :T], attn)
        z_p = m_p_exp + np.random.randn(1, 256, Ty).astype(np.float32) * np.exp(logs_p_exp) * noise_scale

        # Flow
        flow_max = self._flow_eng.get_tensor_profile_shape("z_p", 0)[2][2]
        if z_p.shape[2] > flow_max:
            log.warning("[tts] z_p len %d exceeds TRT flow max %d, using ORT fallback", z_p.shape[2], flow_max)
            if not hasattr(self, '_flow_ort'):
                import onnxruntime as _ort
                self._flow_ort = _ort.InferenceSession(os.path.join(self._trt_dir, "flow.onnx"), providers=["CPUExecutionProvider"])
            z = self._flow_ort.run(None, {"z_p": z_p.astype(np.float32), "y_mask": y_mask.astype(np.float32)})[0]
        else:
            z, = self._trt_run(self._flow_eng, {"z_p": z_p, "y_mask": y_mask}, ["z"])

        # Decoder
        dec_max = self._dec_eng.get_tensor_profile_shape("z", 0)[2][2]
        if z.shape[2] > dec_max:
            log.warning("[tts] z len %d exceeds TRT decoder max %d, using ORT fallback", z.shape[2], dec_max)
            if not hasattr(self, '_dec_ort'):
                import onnxruntime as _ort
                self._dec_ort = _ort.InferenceSession(os.path.join(self._trt_dir, "decoder_spec.onnx"), providers=["CPUExecutionProvider"])
            spec, phase = self._dec_ort.run(None, {"z": z.astype(np.float32)})
        else:
            spec, phase = self._trt_run(self._dec_eng, {"z": z}, ["spec", "phase"])

        # iSTFT (bincount overlap-add)
        tf = np.fft.irfft(spec * np.exp(1j * phase), n=self._n_fft, axis=1)
        wf2d = tf[0] * self._window[0]
        _, T_frames = wf2d.shape
        out_len = (T_frames - 1) * self._hop + self._n_fft
        base = np.arange(T_frames, dtype=np.int32) * self._hop
        offsets = np.arange(self._n_fft, dtype=np.int32)[:, None]
        indices = (base + offsets).ravel()
        audio = np.bincount(indices, weights=wf2d.ravel().astype(np.float64), minlength=out_len).astype(np.float32)
        audio = audio[self._n_fft // 2:out_len - self._n_fft // 2].reshape(1, -1) * self._gain

        if self._speed and self._speed != 1.0:
            target_len = int(audio.shape[1] / self._speed)
            xp = np.linspace(0, 1, audio.shape[1])
            x = np.linspace(0, 1, target_len)
            audio = np.interp(x, xp, audio[0]).reshape(1, -1).astype(np.float32)

        audio_f32 = audio[0]
        peak = max(abs(audio_f32.max()), abs(audio_f32.min()), 1.0)
        if peak > 1.5: audio_f32 = audio_f32 * (1.0 / peak)
        return audio_f32

    def synthesize_stream(self, text: str):
        chunks = self._split_sentences(text)
        for chunk_text in chunks:
            audio_f32 = self._synthesize_chunk(chunk_text)
            pcm = np.clip(audio_f32 * 32767, -32768, 32767).astype(np.int16).tobytes()
            for i in range(0, len(pcm), CHUNK_BYTES):
                yield pcm[i:i + CHUNK_BYTES]


def _build_tts_adapter(cfg: dict) -> TTSAdapter:
    import os
    backend = cfg.get('backend', 'sherpa-onnx')
    model_dir = cfg.get('model_dir', '/models/sherpa-onnx/tts')
    speaker_id = int(cfg.get('speaker_id', 0))
    speed = float(cfg.get('speed', 1.0))

    if backend == 'trt':
        trt_dir = cfg.get('trt_dir', os.path.join(model_dir, 'trt'))
        model_type = cfg.get('model_type', 'v8')
        return TRTTSAdapter(model_dir, trt_dir, speaker_id, speed, model_type)

    return SherpaOnnxTTSAdapter(model_dir, speaker_id, speed)


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class _TTSNode(Node):
    def __init__(self, input_topic: Optional[str], adapter: Optional[TTSAdapter], node_suffix: str = ''):
        node_name = f"tts_{node_suffix}" if node_suffix else "tts"
        super().__init__(node_name)
        self._input_topic  = input_topic or ''
        self._output_topic = f"{input_topic}/tts" if input_topic else '/perception/tts'
        self._adapter      = adapter
        self.state         = "idle"
        self._text_queue   = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event   = threading.Event()
        from audio_msgs.msg import AudioChunk
        self._pub = self.create_publisher(AudioChunk, self._output_topic, _LOW_LAT_QOS)
        self._perf_pub = self.create_publisher(String, '/perception/perf_spans', _LOW_LAT_QOS)
        if input_topic:
            self._sub = self.create_subscription(String, self._input_topic, self._text_cb, _LOW_LAT_QOS)
        else:
            self._sub = None
        log.info(f"[tts] node created: subscribing={self._input_topic or '(none)'}, publishing={self._output_topic}")

    def start(self) -> dict:
        while not self._text_queue.empty():
            try: self._text_queue.get_nowait()
            except Exception: break
        if self.state == "running":
            return self._status_dict()
        if not self._adapter:
            raise RuntimeError("TTS adapter not configured")
        # Dry-run: verify model can synthesize before declaring running
        try:
            test_chunks = list(self._adapter.synthesize_stream("."))
            if not test_chunks:
                return {"state": "error", "message": "TTS dry-run produced no audio"}
        except Exception as e:
            return {"state": "error", "message": f"TTS dry-run failed: {e}"}
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self.state = "running"
        return self._status_dict()

    def stop(self) -> dict:
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)
        self.state = "idle"
        return {"state": "idle"}

    def enqueue(self, text: str, trace_id: str = ''):
        if self.state != "running":
            raise RuntimeError("TTS not running; call start first")
        self._text_queue.put((text, trace_id))

    def _text_cb(self, msg: String):
        if self.state != "running": return
        try:
            text = json.loads(msg.data).get("text","")
        except Exception:
            text = msg.data.strip()
        if text:
            log.info(f"[tts] received text from topic: {text[:50]}...")
            self._text_queue.put((text, ''))

    def _worker(self):
        from audio_msgs.msg import AudioChunk
        import time as _time

        # Real-time pacing: publish frames at playback rate to avoid bursts/gaps
        FRAME_DURATION = CHUNK_BYTES / (SAMPLE_RATE * 2)  # 0.1s per 3200-byte frame
        PREBUF_FRAMES  = 3  # buffer 3 frames (~300ms) before starting real-time pacing

        while not self._stop_event.is_set():
            try:
                item = self._text_queue.get(timeout=1)
            except queue.Empty:
                continue
            # Unpack queue item: (text, trace_id) or plain text for backward compat
            if isinstance(item, tuple):
                text, _trace_id = item
            else:
                text, _trace_id = item, ''
            try:
                import time as _time
                t_start = _time.monotonic()
                t_start_wall = _time.time()  # wall-clock for perf span
                t0_wall = None  # wall-clock when playback starts (prebuf complete)
                total = 0
                buf   = b''
                t0    = None  # wall-clock start of playback
                frames_sent = 0
                prebuf = []   # pre-buffer queue

                for raw_chunk in self._adapter.synthesize_stream(text):
                    if self._stop_event.is_set():
                        break
                    buf  += raw_chunk
                    total += len(raw_chunk)
                    # split into CHUNK_BYTES frames
                    while len(buf) >= CHUNK_BYTES:
                        frame = buf[:CHUNK_BYTES]
                        buf   = buf[CHUNK_BYTES:]

                        # Pre-buffer phase: accumulate a few frames before pacing
                        if t0 is None:
                            prebuf.append(frame)
                            if len(prebuf) >= PREBUF_FRAMES:
                                # Flush pre-buffer and start real-time clock
                                t0 = _time.monotonic()
                                t0_wall = _time.time()
                                for pf in prebuf:
                                    msg = AudioChunk()
                                    msg.header.stamp = self.get_clock().now().to_msg()
                                    msg.format = "audio/pcm-16k"
                                    msg.data   = list(pf)
                                    self._pub.publish(msg)
                                    frames_sent += 1
                                prebuf = []
                            continue

                        # Real-time pacing
                        target = t0 + frames_sent * FRAME_DURATION
                        now = _time.monotonic()
                        if now < target:
                            _time.sleep(target - now)
                        msg = AudioChunk()
                        msg.header.stamp = self.get_clock().now().to_msg()
                        msg.format = "audio/pcm-16k"
                        msg.data   = list(frame)
                        self._pub.publish(msg)
                        frames_sent += 1

                # Flush any remaining pre-buffer (short utterances < PREBUF_FRAMES)
                if prebuf and not self._stop_event.is_set():
                    t0 = _time.monotonic()
                    for pf in prebuf:
                        msg = AudioChunk()
                        msg.header.stamp = self.get_clock().now().to_msg()
                        msg.format = "audio/pcm-16k"
                        msg.data   = list(pf)
                        self._pub.publish(msg)
                        frames_sent += 1

                # flush remainder
                if buf and not self._stop_event.is_set():
                    if t0 is not None:
                        target = t0 + frames_sent * FRAME_DURATION
                        now = _time.monotonic()
                        if now < target:
                            _time.sleep(target - now)
                    msg = AudioChunk()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.format = "audio/pcm-16k"
                    msg.data   = list(buf)
                    self._pub.publish(msg)
                log.info(f"[tts] spoke {len(text)} chars → {total} bytes ({frames_sent} frames) in {_time.monotonic() - t_start:.2f}s")
                # 上报 TTS perf spans（生成 + 播放）
                try:
                    import json as _json
                    t_end_wall = _time.time()
                    spans = []
                    _span_base = {"type": "perf_span", "component": "perception"}
                    if _trace_id:
                        _span_base["trace_id"] = _trace_id
                    if t0_wall:
                        spans.append({**_span_base, "span": "tts_generate",
                                      "start_ts": t_start_wall, "end_ts": t0_wall,
                                      "meta": {"chars": len(text)}})
                        spans.append({**_span_base, "span": "tts_playback",
                                      "start_ts": t0_wall, "end_ts": t_end_wall,
                                      "meta": {"frames": frames_sent}})
                    else:
                        # 没有 prebuf（极短文本），合并为一个 span
                        spans.append({**_span_base, "span": "tts_generate",
                                      "start_ts": t_start_wall, "end_ts": t_end_wall,
                                      "meta": {"chars": len(text), "frames": frames_sent}})
                    for sp in spans:
                        perf_msg = String()
                        perf_msg.data = _json.dumps(sp)
                        self._perf_pub.publish(perf_msg)
                except Exception:
                    pass
            except Exception as e:
                log.error(f"[tts] synthesis error: {e}", exc_info=True)

    def _status_dict(self) -> dict:
        return {
            "state":     self.state,
            "topic_in":  [{"topic": self._input_topic,  "format": "data/json",     "desc": "text to synthesize"}],
            "topic_out": [{"topic": self._output_topic, "format": "audio/pcm-16k", "desc": "synthesized PCM audio"}],
        }


# ── Plugin ────────────────────────────────────────────────────────────────────

class TTSPlugin:
    PREFIX = "tts"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg      = plugin_cfg
        self._loading  = False
        self._load_error = None
        try:
            self._adapter  = _build_tts_adapter(plugin_cfg)
        except Exception as e:
            log.error(f"[tts] failed to load model: {e}", exc_info=True)
            self._adapter = None
            self._load_error = str(e)
        self._nodes: dict[str, _TTSNode] = {}
        self._executor = executor
        log.info(f"[tts] plugin init: sherpa-onnx VITS, "
                 f"speaker_id={plugin_cfg.get('speaker_id', 0)}, speed={plugin_cfg.get('speed', 1.0)}")

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "tts" else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            if self._loading:
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": "loading",
                    "desc": "Downloading TTS model...",
                }
            if self._load_error:
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": "error",
                    "desc": f"Model load failed: {self._load_error}",
                }
            input_topic = args.get("input_topic", "")
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": node.state,
                    "topic_in":  [{"topic": node._input_topic,  "format": "data/json",     "desc": ""}],
                    "topic_out": [{"topic": node._output_topic, "format": "audio/pcm-16k", "desc": ""}],
                    "desc": "TTS service — converts text to audio/pcm-16k",
                }
            if instance_id:
                # Instance requested but not running — return inferred topics for this instance only.
                inferred_out = f"{input_topic}/tts" if input_topic else "/perception/tts"
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": "idle",
                    "topic_in":  [{"topic": input_topic,  "format": "data/json",     "desc": ""}] if input_topic else [],
                    "topic_out": [{"topic": inferred_out, "format": "audio/pcm-16k", "desc": ""}],
                    "desc": "TTS service — converts text to audio/pcm-16k",
                }
            # Aggregate info (no instance_id = ping/overview only)
            if self._nodes:
                topics_in = [{"topic": n._input_topic, "format": "data/json", "desc": ""} for n in self._nodes.values()]
                topics_out = [{"topic": n._output_topic, "format": "audio/pcm-16k", "desc": ""} for n in self._nodes.values()]
                states = list(set(n.state for n in self._nodes.values()))
                state = "running" if "running" in states else states[0] if states else "idle"
            else:
                inferred_out = f"{input_topic}/tts" if input_topic else "/perception/tts"
                topics_in = [{"topic": input_topic, "format": "data/json", "desc": ""}]
                topics_out = [{"topic": inferred_out, "format": "audio/pcm-16k", "desc": ""}]
                state = "idle"
            return {
                "name": "TTS", "manufacture": "Embodied", "model": "tts",
                "state": state,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "TTS service — converts text to audio/pcm-16k",
            }

        elif action == "start":
            if self._loading:
                return {"state": "loading", "message": "TTS model is being downloaded, please wait..."}
            if self._load_error:
                return {"state": "error", "message": f"TTS model failed to load: {self._load_error}"}
            if not self._adapter:
                return {"state": "error", "message": "TTS model not loaded"}
            input_topic = args.get("input_topic") or ''
            node_key = instance_id or input_topic or '_default'
            # Clean up _default node if it would conflict with this instance
            if '_default' in self._nodes and node_key != '_default':
                default_node = self._nodes['_default']
                if default_node._input_topic == input_topic or default_node._output_topic == (f"{input_topic}/tts" if input_topic else '/perception/tts'):
                    default_node.stop()
                    self._executor.remove_node(default_node)
                    del self._nodes['_default']
            if node_key not in self._nodes:
                node = _TTSNode(input_topic or None, self._adapter,
                                node_suffix=node_key.replace('/', '_').replace('-', '_'))
                self._executor.add_node(node)
                self._nodes[node_key] = node
            elif input_topic and self._nodes[node_key]._input_topic != input_topic:
                # Input topic changed for existing instance — recreate
                old_node = self._nodes[node_key]
                old_node.stop()
                self._executor.remove_node(old_node)
                node = _TTSNode(input_topic, self._adapter,
                                node_suffix=node_key.replace('/', '_').replace('-', '_'))
                self._executor.add_node(node)
                self._nodes[node_key] = node
            return self._nodes[node_key].start()

        elif action == "stop":
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                result = node.stop()
                self._executor.remove_node(node)
                del self._nodes[instance_id]
                return result
            elif not instance_id and self._nodes:
                for key in list(self._nodes.keys()):
                    self._nodes[key].stop()
                    self._executor.remove_node(self._nodes[key])
                    del self._nodes[key]
                return {"state": "idle"}
            return {"state": "idle"}

        elif action == "speak":
            if self._loading:
                return {"state": "loading", "message": "TTS model is being downloaded, please wait..."}
            if self._load_error or not self._adapter:
                return {"state": "error", "message": f"TTS model not available: {self._load_error or 'not loaded'}"}
            text = args.get("text", "")
            if not text:
                raise ValueError("text is required")
            # Find any existing running node to reuse
            node = None
            for n in self._nodes.values():
                if n.state == "running":
                    node = n
                    break
            if node is None:
                # No running node — use instance key or fallback
                node_key = instance_id or '_default'
                if node_key not in self._nodes:
                    input_topic = args.get("input_topic") or None
                    adapter = self._adapter
                    if instance_id and instance_id in self._instance_configs:
                        inst_adapter = _build_tts_adapter(self._instance_configs[instance_id])
                        if inst_adapter:
                            adapter = inst_adapter
                    node = _TTSNode(input_topic, adapter,
                                    node_suffix=node_key.replace('/', '_').replace('-', '_'))
                    self._executor.add_node(node)
                    self._nodes[node_key] = node
                else:
                    node = self._nodes[node_key]
                if node.state != "running":
                    node.start()
            node.enqueue(text, trace_id=args.get('_trace_id', ''))
            return {"status": "queued", "text": text}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v}
            # Update config and rebuild adapter
            if 'speaker_id' in cfg:
                self._cfg['speaker_id'] = int(cfg['speaker_id'])
            if 'speed' in cfg:
                self._cfg['speed'] = float(cfg['speed'])
            self._adapter = _build_tts_adapter(self._cfg)
            # Stop all nodes (they'll use new adapter on next start)
            for key in list(self._nodes.keys()):
                self._nodes[key].stop()
                self._executor.remove_node(self._nodes[key])
                del self._nodes[key]
            return {"status": "configured"}

        return None

    def synthesize_raw(self, text: str) -> bytes:
        """Synthesize text and return raw PCM bytes (16kHz 16-bit mono)."""
        if not self._adapter:
            raise RuntimeError("TTS adapter not configured")
        return self._adapter.synthesize(text)
