#!/usr/bin/env python3
"""
plugins/tts.py — TTSPlugin: sherpa-onnx VITS TTS.

On-device text-to-speech using sherpa-onnx MeloTTS (Chinese + English).
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_BYTES = 3200  # 100ms @ 16kHz 16-bit mono

# EOF magic: 8 bytes (4 samples [1, -1, 1, -1])，标记 utterance 结束
# 正常 chunk 始终 3200 bytes，8 bytes 短 chunk 不会被误判
# 即使被不识别 EOF 的旧 Speaker 播放，也只是 0.25ms 极微弱交流声
AUDIO_EOF_MAGIC = b'\x01\x00\xff\xff\x01\x00\xff\xff'

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
                    "enum": ["start", "stop", "speak", "info", "config", "interrupt"],
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
            "required": ["action"],
            "x-completion": {
                "actions": ["speak"],
                "timeout": 60
            },
            "x-hooks": {
                "on_interrupt_speak": {"action": "interrupt"},
            }
        },
        "configSchema": {
            "type": "object",
            "properties": {
                # Engine belongs here, not only in config.yaml: the dashboard
                # builds the config form from configSchema, so an engine that
                # exists solely as a baked YAML key cannot be seen or switched
                # without rebuilding the image. Mirrors asr_model in asr.py.
                "tts_engine": {"type": "string", "enum": ["vits2_trt", "sherpa_onnx"],
                               "description": "TTS engine (vits2_trt = VITS2 TensorRT on Jetson, "
                                              "sherpa_onnx = sherpa-onnx Matcha on CPU)",
                               "default": "vits2_trt", "scope": "shared"},
                "speaker_id": {"type": "integer", "description": "Speaker ID (VITS2 supports 0 only)", "default": 0, "scope": "shared"},
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




def _build_tts_adapter(cfg: dict) -> TTSAdapter:
    import os
    backend = cfg.get('backend', 'sherpa-onnx')
    model_dir = cfg.get('model_dir', '/models/sherpa-onnx/tts')
    speaker_id = int(cfg.get('speaker_id', 0))
    speed = float(cfg.get('speed', 1.0))

    log.info(f"[tts] _build_tts_adapter: backend={backend}, model_dir={model_dir}, "
             f"speaker_id={speaker_id}, speed={speed}")

    if backend == 'trt':
        trt_dir = cfg.get('trt_dir', os.path.join(model_dir, 'trt'))
        model_type = cfg.get('model_type', 'mel20full_d50')
        log.info(f"[tts] _build_tts_adapter: -> TRT backend (trt_dir={trt_dir}, model_type={model_type})")
        return TRTTSAdapter(model_dir, trt_dir, speaker_id, speed, model_type)

    log.info("[tts] _build_tts_adapter: -> sherpa-onnx backend")
    return SherpaOnnxTTSAdapter(model_dir, speaker_id, speed)


# ── ORT Session Cache ────────────────────────────────────────────────────────
_ORT_SESSIONS = {}
_TRT_ENGINES = {}


def _get_ort_session(path):
    if path not in _ORT_SESSIONS:
        import onnxruntime as ort
        _ORT_SESSIONS[path] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _ORT_SESSIONS[path]


_TRT_NP_DTYPE = None


def _trt_np_dtype(trt, dt):
    """TRT DataType → numpy dtype.

    不用 trt.nptype()：TRT 8.5 的实现里引用了 np.bool，而 numpy>=1.24 已移除该别名，
    JP5 镜像上一调用就 AttributeError。
    """
    global _TRT_NP_DTYPE
    if _TRT_NP_DTYPE is None:
        m = {trt.DataType.FLOAT: np.float32, trt.DataType.HALF: np.float16,
             trt.DataType.INT32: np.int32, trt.DataType.INT8: np.int8,
             trt.DataType.BOOL: np.bool_}
        for attr, npt in (("UINT8", np.uint8), ("INT64", np.int64)):
            d = getattr(trt.DataType, attr, None)
            if d is not None:
                m[d] = npt
        _TRT_NP_DTYPE = m
    try:
        return _TRT_NP_DTYPE[dt]
    except KeyError:
        raise RuntimeError(f"unsupported TRT tensor dtype: {dt}")


def _get_trt_engine(path):
    if path not in _TRT_ENGINES:
        import tensorrt as trt
        with open(path, "rb") as f:
            _TRT_ENGINES[path] = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
    return _TRT_ENGINES[path]


# ── TRT TTS Adapter (VITS2-Mix, no PyTorch) ────────────────────────────────

class TRTTSAdapter(TTSAdapter):
    """VITS2-Mix TensorRT TTS adapter — no PyTorch dependency.

    encoder + flow + decoder 全部走 TRT，NumPy 做 MAS/iSTFT。长文本按句切分，
    每个 chunk 始终走 TRT（不回退 ORT）；超长 chunk 再按 phone 数截断分次。
    """

    _MODEL_CFG = {
        "mel20full_d50": {"n_fft": 128, "hop": 4, "gain": 0.0833},
        # mxd_m45d5e6 微调：encoder/flow/dp 与 mel20full_d50 冻结，仅 decoder 不同，
        # 因此 n_fft/hop/gain 完全一致，只换 decoder.trt。
        "mel20full_d50_mxd": {"n_fft": 128, "hop": 4, "gain": 0.0833},
    }

    # 每个 chunk 最大字符数，保证 decoder 输入 Ty 不超过 TRT max shape (JP5=1500)
    MAX_CHUNK_CHARS = 100

    def __init__(self, model_dir: str, trt_dir: str,
                 speaker_id: int = 0, speed: float = 1.0,
                 model_type: str = "mel20full_d50"):
        self._speed = speed
        self._trt_dir = trt_dir

        # ── 按 model_type + TRT 版本选择 engine：JP5=8.5.x / JP6=10.x ──
        import tensorrt as trt
        trt_ver = trt.__version__
        if trt_ver.startswith("8.5"):
            engine_name = f"vits2_trt_{model_type}_jp5"
        elif trt_ver.startswith("10."):
            engine_name = f"vits2_trt_{model_type}_jp6"
        else:
            raise RuntimeError(
                f"VITS2 TRT engine 仅支持 TRT 8.5.x（JP5）或 10.x（JP6），当前为 TRT {trt_ver}"
            )
        log.info(f"[tts] TRT adapter init: TRT={trt_ver}, engine={engine_name}, "
                 f"model_dir={model_dir}, trt_dir={trt_dir}")

        # ── 运行时下载模型资产（不 bake 进镜像：/models 会被 bind-mount 覆盖）──
        from utils.model_downloader import ensure_model
        ensure_model("vits2_mix", model_dir)
        ensure_model(engine_name, trt_dir)
        log.info(f"[tts] TRT adapter init: assets ready (model_dir={model_dir}, trt_dir={trt_dir})")

        # ── Frontend (G2P) ──
        sys.path.insert(0, model_dir)
        from frontend.cleaner import clean_text_mix
        from frontend import cleaned_text_to_sequence_mix
        self._clean_text = clean_text_mix
        self._seq_mix = cleaned_text_to_sequence_mix

        # ── Model-specific iSTFT parameters ──
        if model_type not in self._MODEL_CFG:
            raise ValueError(
                f"Unknown model_type: {model_type!r}; supported: {sorted(self._MODEL_CFG)}"
            )
        mc = self._MODEL_CFG[model_type]
        self._n_fft = mc["n_fft"]
        self._hop = mc["hop"]
        self._gain = mc["gain"]

        # Periodic Hann window (matches TorchSTFT fftbins=True)
        w = np.hanning(self._n_fft + 1)[:self._n_fft].astype(np.float32)
        self._window = w.reshape(1, self._n_fft, 1)

        # ── Encoder（纯 TRT，不回退 ORT）──
        # encoder.trt 的 profile 上限是 ph:1x1000（见 tools/trt 构建命令）；
        # 超过该上限的文本在 _synthesize_chunk 里按词截断、分次编码，绝不回退 ORT。
        enc_trt_path = os.path.join(trt_dir, "encoder.trt")
        if not os.path.exists(enc_trt_path):
            raise RuntimeError(
                f"VITS2 TRT encoder engine not found: {enc_trt_path}. "
                f"请先用 trtexec 从 encoder_duration.onnx（opset 16）构建 encoder.trt"
            )
        self._enc_eng = _get_trt_engine(enc_trt_path)
        if self._enc_eng is None:
            raise RuntimeError(f"Failed to deserialize encoder engine {enc_trt_path}")
        self._enc_max_T = int(self._enc_eng.get_tensor_profile_shape("ph", 0)[2][1])
        log.info("[tts] encoder backend=TRT (%s, max T_phone=%d)",
                 enc_trt_path, self._enc_max_T)

        # ── TRT engines (shared across instances via module cache) ──
        flow_path = os.path.join(trt_dir, "flow.trt")
        dec_path = os.path.join(trt_dir, "decoder.trt")
        self._flow_eng = _get_trt_engine(flow_path)
        self._dec_eng = _get_trt_engine(dec_path)
        if self._flow_eng is None or self._dec_eng is None:
            raise RuntimeError(f"Failed to load TRT engines from {trt_dir}")

        # ── CUDA allocator ──
        self._cuda = ctypes.CDLL("libcudart.so")
        self._cuda.cudaMalloc.restype = int
        self._cuda.cudaMalloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._cuda.cudaFree.restype = int
        self._cuda.cudaFree.argtypes = [ctypes.c_void_p]
        self._cuda.cudaMemcpy.restype = int
        self._cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_size_t, ctypes.c_int]

        # ── 复用的 execution context + buffer（避免每次推理 create/malloc/free）──
        self._ctx_cache = {}   # engine -> execution context
        self._gpu_buf = {}     # (id(engine), name) -> (ptr, size)
        self._cpu_buf = {}     # (id(engine), name) -> np.ndarray
        self._bound_shape = {}  # (id(engine), name) -> tuple  (只在 shape 变化时才 set_input_shape)

        log.info("[tts] TRT adapter loaded: model=%s n_fft=%d hop=%d gain=%.4f",
                 model_type, self._n_fft, self._hop, self._gain)

    def _gpu_alloc(self, size):
        ptr = ctypes.c_void_p(0)
        self._cuda.cudaMalloc(ctypes.byref(ptr), size)
        return ptr

    def _get_ctx(self, engine):
        ctx = self._ctx_cache.get(engine)
        if ctx is None:
            ctx = engine.create_execution_context()
            self._ctx_cache[engine] = ctx
        return ctx

    def _gpu_buf_for(self, engine, name, size):
        """复用 GPU buffer（不够大才重新分配），避免每次推理 malloc/free。"""
        key = (id(engine), name)
        entry = self._gpu_buf.get(key)
        if entry is not None and entry[1] >= size:
            return entry[0]
        if entry is not None:
            self._cuda.cudaFree(entry[0])
        ptr = self._gpu_alloc(size)
        self._gpu_buf[key] = (ptr, size)
        return ptr

    def _cpu_buf_for(self, engine, name, shape, dtype=np.float32):
        """复用 CPU buffer（取前 N 元素 reshape 到目标 shape，避免反复分配）。"""
        key = (id(engine), name, np.dtype(dtype).str)
        arr = self._cpu_buf.get(key)
        need = int(np.prod(shape))
        if arr is None or arr.size < need:
            arr = np.empty(shape, dtype=dtype)
            self._cpu_buf[key] = arr
            return arr
        return arr.ravel()[:need].reshape(shape)

    def _trt_run(self, engine, inputs, output_names):
        import tensorrt as trt
        ctx = self._get_ctx(engine)
        gpu_ptrs, outputs = {}, {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            # encoder 的 ph/to/la/xl 是 int32，flow/decoder 是 float32 —— 按 engine 声明的 dtype 走
            dt = _trt_np_dtype(trt, engine.get_tensor_dtype(name))
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                data = np.ascontiguousarray(inputs[name], dtype=dt)
                key = (id(engine), name)
                if self._bound_shape.get(key) != data.shape:
                    # TRT 10.x 每次 set_input_shape 都会触发 re-optimize（实测 ~1s），
                    # 只有 shape 变化时才需要重新 bind。
                    ctx.set_input_shape(name, data.shape)
                    self._bound_shape[key] = data.shape
                ptr = self._gpu_buf_for(engine, name, data.nbytes)
                self._cuda.cudaMemcpy(ptr, data.ctypes.data, data.nbytes, 1)  # H2D
                gpu_ptrs[name] = ptr
            else:
                shape = tuple(ctx.get_tensor_shape(name))
                arr = self._cpu_buf_for(engine, name, shape, dt)
                outputs[name] = arr
                ptr = self._gpu_buf_for(engine, name, arr.nbytes)
                gpu_ptrs[name] = ptr
        bindings = [gpu_ptrs[engine.get_tensor_name(i)].value
                     for i in range(engine.num_io_tensors)]
        ctx.execute_v2(bindings)
        for name, arr in outputs.items():
            self._cuda.cudaMemcpy(arr.ctypes.data, gpu_ptrs[name], arr.nbytes, 2)  # D2H
        return tuple(outputs[n] for n in output_names)

    @staticmethod
    def _split_sentences(text: str) -> list:
        """按句号截断，超长文本分批，每批不超过 MAX_CHUNK_CHARS 字符。

        单个无标点句子超过上限时也会切成多个片段，保证 decoder Ty 不超过
        TRT engine 的 max shape（JP5 约 1500）。
        """
        import re
        sentences = re.split(r'(?<=[。！？\n])', text)
        limit = TRTTSAdapter.MAX_CHUNK_CHARS
        chunks, buf = [], ""

        def flush():
            nonlocal buf
            if buf:
                chunks.append(buf)
                buf = ""

        for s in sentences:
            s = s.strip()
            if not s:
                continue
            while s:
                room = limit - len(buf)
                if room >= len(s):
                    buf += s
                    break
                if room > 0:
                    buf += s[:room]
                    s = s[room:]
                flush()
                # 此时 buf 为空；若 s 仍超长，切成完整片段
                while len(s) >= limit:
                    chunks.append(s[:limit])
                    s = s[limit:]
        flush()
        return chunks if chunks else [text]

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def _encode_phones(self, text: str):
        """Text → interleaved phone/tone/lang ID 序列，返回 (ph, to, la, T)。"""
        _, phones, tones, langs, _ = self._clean_text(text)
        phone_ids, tone_ids, lang_ids = self._seq_mix(phones, tones, langs)
        phone_ids = [0] + [p for pid in phone_ids for p in (pid, 0)]
        tone_ids = [0] + [t for tid in tone_ids for t in (tid, 0)]
        lang_ids = [0] + [l for lid in lang_ids for l in (lid, 0)]
        T = len(phone_ids)
        return (np.array([phone_ids], dtype=np.int32),
                np.array([tone_ids], dtype=np.int32),
                np.array([lang_ids], dtype=np.int32),
                np.array([T], dtype=np.int32))

    def _split_text_under_phones(self, text: str) -> list:
        """把一段文本按词贪心拆成多段，保证每段的 phone 数 ≤ _enc_max_T。

        只在极少数超长/高音素密度文本（如整段英文）触发；正常中文 MAX_CHUNK_CHARS=100
        对应 T≈400，远低于 encoder.trt 的 1000 上限，不会走到这里。
        """
        import re
        limit = self._enc_max_T
        tokens = re.findall(r"\S+\s*", text) or [text]
        pieces, buf = [], ""
        for tok in tokens:
            cand = buf + tok
            if buf and self._encode_phones(cand)[3].item() > limit:
                pieces.append(buf)
                buf = tok
            else:
                buf = cand
        if buf:
            pieces.append(buf)
        # 兜底：仍超限的片段（如无空格的超长数字/URL）按实际 phone 数递归二分，
        # 绝不按字符数硬切 —— 1 字符 ≈ 2 phone，按字符切会仍然超限。
        out = []
        for p in pieces:
            out.extend(self._hard_split_under_phones(p))
        return out or [text]

    def _hard_split_under_phones(self, text: str) -> list:
        """递归二分，直到每段编码后的 phone 数 ≤ _enc_max_T。"""
        if self._encode_phones(text)[3].item() <= self._enc_max_T:
            return [text]
        mid = len(text) // 2
        if mid == 0:
            return [text]  # 单字符仍超限（几乎不可能），交给 encoder 暴露错误
        return self._hard_split_under_phones(text[:mid]) + self._hard_split_under_phones(text[mid:])

    def _synth_phones(self, ph, to, la, xl, T, noise_scale):
        """phone ID 序列 → audio float32（encoder + flow + decoder + iSTFT，全 TRT）。"""
        # ── 1. Encoder (TRT) ──
        m_p, logs_p, logw, x_mask = self._trt_run(
            self._enc_eng, {"ph": ph, "to": to, "la": la, "xl": xl},
            ("m_p", "logs_p", "logw", "x_mask"))

        # ── 2. Duration expansion (pure NumPy MAS) ──
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
                if hi > lo:
                    attn[0, tx, lo:hi] = 1.0

        m_p_exp = np.matmul(m_p[:, :, :T], attn)
        logs_p_exp = np.matmul(logs_p[:, :, :T], attn)
        z_p = m_p_exp + np.random.randn(1, 256, Ty).astype(np.float32) * np.exp(logs_p_exp) * noise_scale

        # ── 3. Flow (TRT) ──
        z, = self._trt_run(self._flow_eng, {"z_p": z_p, "y_mask": y_mask}, ["z"])

        # ── 4. Decoder (TRT) ──
        spec, phase = self._trt_run(self._dec_eng, {"z": z}, ["spec", "phase"])

        # ── 5. iSTFT (NumPy bincount overlap-add) ──
        tf = np.fft.irfft(spec * np.exp(1j * phase), n=self._n_fft, axis=1)
        wf2d = tf[0] * self._window[0]
        _, T_frames = wf2d.shape
        out_len = (T_frames - 1) * self._hop + self._n_fft
        base = np.arange(T_frames, dtype=np.int32) * self._hop
        offsets = np.arange(self._n_fft, dtype=np.int32)[:, None]
        indices = (base + offsets).ravel()
        audio = np.bincount(indices, weights=wf2d.ravel().astype(np.float64),
                            minlength=out_len).astype(np.float32)
        audio = audio[self._n_fft // 2:out_len - self._n_fft // 2].reshape(1, -1) * self._gain

        if self._speed and self._speed != 1.0:
            target_len = int(audio.shape[1] / self._speed)
            xp = np.linspace(0, 1, audio.shape[1])
            x = np.linspace(0, 1, target_len)
            audio = np.interp(x, xp, audio[0]).reshape(1, -1).astype(np.float32)

        return audio[0]

    def _synthesize_chunk(self, text: str, noise_scale: float = 0.667):
        """Synthesize a single text chunk through TRT, returning float32 numpy array.

        若 phone 序列超过 encoder.trt 的 max profile，则按词截断、分次编码后拼接，
        绝不回退 ORT。
        """
        t0 = time.perf_counter()
        ph, to, la, xl = self._encode_phones(text)
        T = int(xl.item())

        if T <= self._enc_max_T:
            audio_f32 = self._synth_phones(ph, to, la, xl, T, noise_scale)
        else:
            log.info("[tts] T_phone=%d exceeds encoder max %d, splitting text into pieces",
                     T, self._enc_max_T)
            audios = []
            for piece in self._split_text_under_phones(text):
                pph, pto, pla, pxl = self._encode_phones(piece)
                audios.append(self._synth_phones(pph, pto, pla, pxl, int(pxl.item()), noise_scale))
            audio_f32 = np.concatenate(audios)

        peak = max(abs(audio_f32.max()), abs(audio_f32.min()), 1.0)
        if peak > 1.5:
            audio_f32 = audio_f32 * (1.0 / peak)
        t1 = time.perf_counter()
        log.info("[tts] chunk %d chars %d phones → audio=%.2fs | total=%dms",
                 len(text), T, len(audio_f32) / 16000, int((t1 - t0) * 1000))
        return audio_f32

    def synthesize_stream(self, text: str):
        chunks = self._split_sentences(text)
        for chunk_text in chunks:
            audio_f32 = self._synthesize_chunk(chunk_text)
            pcm = np.clip(audio_f32 * 32767, -32768, 32767).astype(np.int16).tobytes()
            for i in range(0, len(pcm), CHUNK_BYTES):
                yield pcm[i:i + CHUNK_BYTES]


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
        self._interrupt_flag = threading.Event()  # 打断标志：设置后立即停止当前 utterance
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

    def interrupt(self) -> dict:
        """立即中止当前播放：清空队列 + 设置 interrupt flag 让 worker 停止当前 utterance。"""
        # 清空待播放队列
        cleared = 0
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        # 设置 interrupt flag（worker 在每个 frame 前检查）
        self._interrupt_flag.set()
        log.info(f"[tts] interrupted: cleared {cleared} queued item(s)")
        return {"status": "interrupted", "cleared": cleared}

    def enqueue(self, text: str, trace_id: str = '', action_id: str = ''):
        if self.state != "running":
            raise RuntimeError("TTS not running; call start first")
        # 分段：超过 280 字按标点切分，避免超长合成导致延迟或失败
        segments = self._split_text(text, max_chars=280)
        if len(segments) <= 1:
            self._text_queue.put((text, trace_id, action_id))
        else:
            # 只有最后一段带 action_id（触发 ACP callback）
            for i, seg in enumerate(segments):
                is_last = (i == len(segments) - 1)
                self._text_queue.put((seg, trace_id, action_id if is_last else ''))
            log.info(f"[tts] split {len(text)} chars into {len(segments)} segments")

    @staticmethod
    def _split_text(text: str, max_chars: int = 280) -> list:
        """按标点分段，每段不超过 max_chars 字。"""
        import re as _re
        sentences = _re.split(r'(?<=[。！？；\n])', text)
        segments = []
        current = ""
        for sent in sentences:
            if not sent:
                continue
            if len(current) + len(sent) > max_chars and current:
                segments.append(current)
                current = sent
            else:
                current += sent
        if current:
            segments.append(current)
        return segments if segments else [text]

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

        # Real-time pacing: publish frames at playback rate to avoid bursts/gaps.
        # 默认关闭节流（burst 发布）；需要模拟实时播放时设 TTS_PACING=1。
        _pacing = os.environ.get("TTS_PACING", "0").strip().lower() not in ("0", "false", "no", "off")
        FRAME_DURATION = CHUNK_BYTES / (SAMPLE_RATE * 2)  # 0.1s per 3200-byte frame
        PREBUF_FRAMES  = 3 if _pacing else 0  # 关闭节流时无需预缓冲

        while not self._stop_event.is_set():
            try:
                item = self._text_queue.get(timeout=1)
            except queue.Empty:
                continue
            # Unpack queue item: (text, trace_id, action_id) or legacy formats
            if isinstance(item, tuple):
                if len(item) == 3:
                    text, _trace_id, _action_id = item
                elif len(item) == 2:
                    text, _trace_id = item
                    _action_id = ''
                else:
                    text, _trace_id, _action_id = str(item[0]), '', ''
            else:
                text, _trace_id, _action_id = item, '', ''
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
                    if self._stop_event.is_set() or self._interrupt_flag.is_set():
                        break
                    buf  += raw_chunk
                    total += len(raw_chunk)
                    # split into CHUNK_BYTES frames
                    while len(buf) >= CHUNK_BYTES:
                        frame = buf[:CHUNK_BYTES]
                        buf   = buf[CHUNK_BYTES:]

                        # Check interrupt before publishing each frame
                        if self._interrupt_flag.is_set():
                            break

                        # Pre-buffer phase: accumulate a few frames before pacing
                        if t0 is None:
                            prebuf.append(frame)
                            if len(prebuf) >= PREBUF_FRAMES:
                                # Flush pre-buffer and start real-time clock
                                t0 = _time.monotonic()
                                t0_wall = _time.time()
                                # Fire on_speaking hook (playback starting)
                                try:
                                    import urllib.request as _ureq
                                    import json as _jhook
                                    _hreq = _ureq.Request(
                                        "https://localhost:15678/api/hooks/fire",
                                        data=_jhook.dumps({"hook": "on_speaking"}).encode(),
                                        headers={"Content-Type": "application/json"},
                                        method="POST"
                                    )
                                    import ssl as _ssl
                                    _sctx = _ssl.create_default_context()
                                    _sctx.check_hostname = False
                                    _sctx.verify_mode = _ssl.CERT_NONE
                                    _ureq.urlopen(_hreq, timeout=2, context=_sctx)
                                except Exception:
                                    pass
                                for pf in prebuf:
                                    msg = AudioChunk()
                                    msg.header.stamp = self.get_clock().now().to_msg()
                                    msg.format = "audio/pcm-16k"
                                    msg.data   = list(pf)
                                    self._pub.publish(msg)
                                    frames_sent += 1
                                prebuf = []
                            continue

                        # Real-time pacing（TTS_PACING=0 时跳过，直接 burst 发布）
                        if _pacing:
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
                if prebuf and not self._stop_event.is_set() and not self._interrupt_flag.is_set():
                    t0 = _time.monotonic()
                    for pf in prebuf:
                        msg = AudioChunk()
                        msg.header.stamp = self.get_clock().now().to_msg()
                        msg.format = "audio/pcm-16k"
                        msg.data   = list(pf)
                        self._pub.publish(msg)
                        frames_sent += 1

                # flush remainder
                if buf and not self._stop_event.is_set() and not self._interrupt_flag.is_set():
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

                # Clear interrupt flag after utterance is done (interrupted or complete)
                if self._interrupt_flag.is_set():
                    self._interrupt_flag.clear()
                    log.info(f"[tts] utterance interrupted after {frames_sent} frames")
                else:
                    log.info(f"[tts] spoke {len(text)} chars → {total} bytes ({frames_sent} frames) in {_time.monotonic() - t_start:.2f}s")

                # 发布 EOF 标记：告知下游 Speaker 当前 utterance 已结束
                self._publish_eof()
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

                # ACP: 推送动作完成回调到 Agent Core
                # Also fire on_idle hook (LED off immediately after playback)
                if _action_id:
                    try:
                        import urllib.request as _urllib
                        import ssl as _ssl
                        import os as _os
                        _agent_core_url = _os.environ.get("AGENT_CORE_URL", "https://localhost:15678")
                        _ctx = _ssl.create_default_context()
                        _ctx.check_hostname = False
                        _ctx.verify_mode = _ssl.CERT_NONE
                        # Fire on_idle to turn off LED
                        _idle_req = _urllib.Request(
                            f"{_agent_core_url}/api/hooks/fire",
                            data=json.dumps({"hook": "on_idle"}).encode(),
                            headers={"Content-Type": "application/json"},
                            method="POST"
                        )
                        _urllib.urlopen(_idle_req, timeout=2, context=_ctx)
                        was_interrupted = self._interrupt_flag.is_set()
                        _payload = json.dumps({
                            "action_id": _action_id,
                            "status": "cancelled" if was_interrupted else "completed",
                            "result": {"text": text[:100], "frames": frames_sent},
                        }).encode()
                        _req = _urllib.Request(
                            f"{_agent_core_url}/api/acp/complete",
                            data=_payload,
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        _urllib.urlopen(_req, timeout=3, context=_ctx)
                        log.info(f"[tts] ACP complete: {_action_id} ({'cancelled' if was_interrupted else 'completed'})")
                    except Exception as e:
                        log.warning(f"[tts] ACP callback failed: {e}")
            except Exception as e:
                log.error(f"[tts] synthesis error: {e}", exc_info=True)

    def _status_dict(self) -> dict:
        return {
            "state":     self.state,
            "topic_in":  [{"topic": self._input_topic,  "format": "data/json",     "desc": "text to synthesize"}],
            "topic_out": [{"topic": self._output_topic, "format": "audio/pcm-16k", "desc": "synthesized PCM audio"}],
        }

    def _publish_eof(self):
        """发布 EOF magic chunk，标记当前 utterance 结束。"""
        from audio_msgs.msg import AudioChunk
        msg = AudioChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "audio/pcm-16k"
        msg.data = list(AUDIO_EOF_MAGIC)
        self._pub.publish(msg)


# ── Plugin ────────────────────────────────────────────────────────────────────

class SherpaOnnxTTSPlugin:
    PREFIX = "tts"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg      = plugin_cfg
        self._loading  = False
        self._load_error = None
        log.info(f"[tts] plugin init: cfg keys={sorted(plugin_cfg.keys())}, "
                 f"backend={plugin_cfg.get('backend', '(unset->sherpa-onnx)')}")
        try:
            self._adapter  = _build_tts_adapter(plugin_cfg)
        except Exception as e:
            log.error(f"[tts] failed to load model: {e}", exc_info=True)
            self._adapter = None
            self._load_error = str(e)
        self._nodes: dict[str, _TTSNode] = {}
        self._instance_configs: dict = {}
        # main.py serves MCP over ThreadingHTTPServer, so start/stop/speak/config
        # can run concurrently. Every read-modify-write of _nodes must hold this:
        # otherwise two threads both pass a "key not in _nodes" check, both build
        # a node, and the dict keeps only the last — leaving the other running but
        # unreachable, with a duplicate publisher on the same topic that nothing
        # can stop. See perception/README.md § Plugin Concurrency.
        # RLock: dispatch paths nest (start → _dispose_node).
        self._nodes_lock = threading.RLock()
        self._executor = executor
        adapter_type = type(self._adapter).__name__ if self._adapter else "None"
        log.info(f"[tts] plugin init: adapter={adapter_type}, backend={plugin_cfg.get('backend','(unset)')}, "
                 f"load_error={self._load_error or '(none)'}, "
                 f"speaker_id={plugin_cfg.get('speaker_id', 0)}, speed={plugin_cfg.get('speed', 1.0)}")

    def _dispose_node(self, node: _TTSNode, key: str = "") -> dict:
        """Stop a node and release its ROS endpoints. Caller holds _nodes_lock.

        destroy_node() matters: without it the publisher and the ROS node name
        outlive the node object, so a later start on the same key collides with a
        still-registered ghost.
        """
        result = {"state": "idle"}
        try:
            result = node.stop()
        except Exception:
            log.error(f"[tts] node.stop() failed while disposing '{key}'", exc_info=True)
        try:
            self._executor.remove_node(node)
        except Exception as error:
            log.warning(f"[tts] failed to remove ROS node '{key}': {error}")
        try:
            node.destroy_node()
        except Exception as error:
            log.warning(f"[tts] failed to destroy ROS node '{key}': {error}")
        return result

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
            # Snapshot under the lock: info is a heartbeat probe and iterating the
            # live dict can raise "dictionary changed size" mid-start.
            with self._nodes_lock:
                node = self._nodes.get(instance_id) if instance_id else None
                nodes_snapshot = list(self._nodes.values())
            if instance_id and node is not None:
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
            if nodes_snapshot:
                topics_in = [{"topic": n._input_topic, "format": "data/json", "desc": ""} for n in nodes_snapshot]
                topics_out = [{"topic": n._output_topic, "format": "audio/pcm-16k", "desc": ""} for n in nodes_snapshot]
                states = list(set(n.state for n in nodes_snapshot))
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
            with self._nodes_lock:
                # Clean up _default node if it would conflict with this instance
                if '_default' in self._nodes and node_key != '_default':
                    default_node = self._nodes['_default']
                    if default_node._input_topic == input_topic or default_node._output_topic == (f"{input_topic}/tts" if input_topic else '/perception/tts'):
                        del self._nodes['_default']
                        self._dispose_node(default_node, '_default')
                node = self._nodes.get(node_key)
                if node is None:
                    node = _TTSNode(input_topic or None, self._adapter,
                                    node_suffix=node_key.replace('/', '_').replace('-', '_'))
                    self._executor.add_node(node)
                    self._nodes[node_key] = node
                elif input_topic and node._input_topic != input_topic:
                    # Input topic changed for existing instance — recreate
                    del self._nodes[node_key]
                    self._dispose_node(node, node_key)
                    node = _TTSNode(input_topic, self._adapter,
                                    node_suffix=node_key.replace('/', '_').replace('-', '_'))
                    self._executor.add_node(node)
                    self._nodes[node_key] = node
                return node.start()

        elif action == "stop":
            with self._nodes_lock:
                if instance_id:
                    node = self._nodes.pop(instance_id, None)
                    if node is None:
                        return {"state": "idle"}
                    return self._dispose_node(node, instance_id)
                for key in list(self._nodes.keys()):
                    self._dispose_node(self._nodes.pop(key), key)
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
            with self._nodes_lock:
                node = None
                for n in self._nodes.values():
                    if n.state == "running":
                        node = n
                        break
                if node is None:
                    # No running node — use instance key or fallback
                    node_key = instance_id or '_default'
                    node = self._nodes.get(node_key)
                    if node is None:
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
                    if node.state != "running":
                        node.start()
            # ACP: 生成 action_id
            import uuid as _uuid
            action_id = f"speak-{_uuid.uuid4().hex[:8]}"
            node.enqueue(text, trace_id=args.get('_trace_id', ''), action_id=action_id)
            return {"status": "queued", "action_id": action_id, "text": text}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v}
            # Update config and rebuild adapter
            if 'speaker_id' in cfg:
                self._cfg['speaker_id'] = int(cfg['speaker_id'])
            if 'speed' in cfg:
                self._cfg['speed'] = float(cfg['speed'])
            self._adapter = _build_tts_adapter(self._cfg)
            # Stop all nodes (they'll use new adapter on next start)
            with self._nodes_lock:
                for key in list(self._nodes.keys()):
                    self._dispose_node(self._nodes.pop(key), key)
            return {"status": "configured"}

        elif action == "interrupt":
            # 立即中止所有 TTS 播放（清空队列 + 停止当前 utterance）
            total_cleared = 0
            interrupted_count = 0
            with self._nodes_lock:
                if instance_id:
                    targets = [self._nodes[instance_id]] if instance_id in self._nodes else []
                else:
                    targets = [n for n in self._nodes.values() if n.state == "running"]
            for node in targets:
                result = node.interrupt()
                total_cleared += result.get('cleared', 0)
                interrupted_count += 1
            return {"status": "interrupted", "nodes": interrupted_count, "cleared": total_cleared}

        return None

    def synthesize_raw(self, text: str) -> bytes:
        """Synthesize text and return raw PCM bytes (16kHz 16-bit mono)."""
        if not self._adapter:
            raise RuntimeError("TTS adapter not configured")
        return self._adapter.synthesize(text)


DEFAULT_TTS_ENGINE = "vits2_trt"
TTS_ENGINES = ("vits2_trt", "sherpa_onnx")
# Where each engine keeps its own model files. Used for any engine other than
# the one config.yaml was written for; see TTSPlugin._model_dir_for.
ENGINE_MODEL_DIRS = {
    "vits2_trt": "/models/vits2",
    "sherpa_onnx": "/models/sherpa-onnx/tts",
}


class TTSPlugin:
    """The single public TTS plugin, delegating to a config-selected engine.

    A facade rather than a `__new__` switch: the engine is a configSchema field,
    so it can change at runtime (`action=config`, `tts_engine=...`) and not only
    at process start. Switching disposes the previous engine's nodes and builds
    the new one in the background — sherpa-onnx downloads its Matcha model in
    its constructor, which would otherwise blow through the 60 s Agent Core
    allows a processor call.
    """

    PREFIX = "tts"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg = dict(plugin_cfg)
        self._executor = executor
        self._lock = threading.Lock()
        self._impl = None
        self._impl_engine = ""
        self._building = ""          # engine name while a build is in flight
        self._build_error = None
        engine = self._select_engine(self._cfg.get("engine")
                                     or self._cfg.get("tts_engine"))
        self._engine = engine
        # model_dir is per engine: sherpa-onnx wants its Matcha/vocoder pair,
        # VITS2 wants its TensorRT release. config.yaml carries one model_dir,
        # written for the engine it also declares — handing that same path to the
        # other engine made sherpa download its models into /models/vits2 and
        # then try to load them from there. So the configured path applies only
        # to the configured engine; every other engine gets its own default.
        self._configured_engine = engine
        self._configured_model_dir = self._cfg.get("model_dir") or ""
        # Built inline at startup so info/start are immediately truthful, and so
        # a misconfigured engine shows up in the boot log rather than on the
        # first utterance. Runtime switches take the background path below.
        try:
            self._impl = self._build(engine)
            self._impl_engine = engine
        except Exception as error:  # noqa: BLE001 - surfaced via info/start
            log.error("[tts] failed to build engine %r: %s", engine, error, exc_info=True)
            self._build_error = str(error)

    # ── engine plumbing ─────────────────────────────────────────────────

    @staticmethod
    def _select_engine(value) -> str:
        engine = str(value or DEFAULT_TTS_ENGINE).strip().lower()
        if engine not in TTS_ENGINES:
            raise ValueError(f"Unsupported TTS engine: {engine}")
        return engine

    def _model_dir_for(self, engine: str) -> str:
        if engine == self._configured_engine and self._configured_model_dir:
            return self._configured_model_dir
        return ENGINE_MODEL_DIRS[engine]

    def _build(self, engine: str):
        cfg = dict(self._cfg)
        cfg["engine"] = engine
        cfg["model_dir"] = self._model_dir_for(engine)
        impl = (self._build_vits2(cfg) if engine == "vits2_trt"
                else SherpaOnnxTTSPlugin(cfg, self._executor))
        # An implementation may swallow its own model-load failure and come back
        # as an object that reports error through info (sherpa does exactly
        # that). Installing it would make the facade claim ready and let a start
        # or a speak "succeed" against a model that never loaded, so ask it.
        state = impl.dispatch("tts", {"action": "info"}) or {}
        if state.get("state") == "error":
            raise RuntimeError(
                state.get("error") or state.get("desc")
                or f"engine {engine} reported an error after construction"
            )
        return impl

    def _build_vits2(self, cfg: dict):
        from plugins.vits2_tts import Vits2TTSPlugin

        return Vits2TTSPlugin(cfg, self._executor)

    def _build_async(self, engine: str) -> None:
        """Build an engine off the request thread; the old one is already gone."""
        def _run():
            try:
                impl = self._build(engine)
            except Exception as error:  # noqa: BLE001
                log.error("[tts] failed to build engine %r: %s", engine, error,
                          exc_info=True)
                with self._lock:
                    if self._building == engine:
                        self._building = ""
                        self._build_error = str(error)
                return
            stale = None
            with self._lock:
                if self._building != engine:
                    stale = impl          # another switch superseded this one
                else:
                    self._impl = impl
                    self._impl_engine = engine
                    self._building = ""
                    self._build_error = None
            if stale is not None:
                _dispose_impl(stale)

        threading.Thread(target=_run, name=f"tts-engine-{engine}", daemon=True).start()

    def get_tools(self) -> list:
        return TOOLS

    # ── dispatch ────────────────────────────────────────────────────────

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "tts" else name
        if action == "config":
            return self._config(name, args)

        with self._lock:
            impl = self._impl
            building = self._building
            error = self._build_error
            engine = self._engine
        if impl is not None:
            result = impl.dispatch(name, args)
            if isinstance(result, dict) and action == "info":
                result.setdefault("engine", engine)
            return result

        # No engine resident: only happens while a switch is building, or after
        # a build failure. Never block the caller waiting for it.
        if action == "info":
            state = "loading" if building else "error"
            result = {
                "name": "TTS", "manufacture": "Embodied", "model": engine,
                "engine": engine, "state": state,
                "desc": (f"Switching to the {building} engine..." if building
                         else f"Engine {engine} failed to load: {error}"),
            }
            if state == "error" and error:
                result["error"] = error
            return result
        if building:
            return {"state": "loading",
                    "message": f"TTS engine {building} is initializing, retry shortly"}
        return {"state": "error", "message": f"TTS engine {engine} failed: {error}"}

    def _config(self, name: str, args: dict) -> dict:
        """Apply shared config, switching engines when tts_engine changes."""
        requested = args.get("tts_engine") or args.get("engine")
        forwarded = {k: v for k, v in args.items()
                     if k not in ("tts_engine", "engine")}
        for key in ("speaker_id", "speed"):
            if key in args:
                self._cfg[key] = args[key]

        if requested:
            engine = self._select_engine(requested)
            with self._lock:
                switching = engine != self._impl_engine or self._impl is None
                if switching:
                    outgoing, self._impl = self._impl, None
                    self._impl_engine = ""
                    self._engine = engine
                    self._cfg["engine"] = engine
                    self._building = engine
                    self._build_error = None
            if switching:
                # Stop the old engine's nodes before the new one publishes on
                # the same topics — two live TTS publishers on one topic is the
                # duplicate-audio failure mode in README § Plugin Concurrency.
                if outgoing is not None:
                    _dispose_impl(outgoing)
                self._build_async(engine)
                log.info("[tts] switching engine to %s", engine)
                return {"status": "configured", "state": "loading",
                        "engine": engine,
                        "message": f"loading the {engine} engine"}

        with self._lock:
            impl = self._impl
            engine = self._engine
        if impl is None:
            return {"status": "configured", "state": "loading", "engine": engine}
        result = impl.dispatch(name, {**forwarded, "action": "config"})
        if isinstance(result, dict):
            result.setdefault("engine", engine)
        return result

    def synthesize_raw(self, text: str) -> bytes:
        """Synthesize text and return raw PCM bytes (16kHz 16-bit mono)."""
        with self._lock:
            impl = self._impl
            engine = self._engine
            error = self._build_error
        if impl is None:
            raise RuntimeError(f"TTS engine {engine} not ready: {error or 'loading'}")
        return impl.synthesize_raw(text)


def _dispose_impl(impl) -> None:
    """Stop every node an engine implementation owns before dropping it."""
    try:
        impl.dispatch("tts", {"action": "stop"})
    except Exception:
        log.error("[tts] failed to stop the outgoing engine", exc_info=True)
