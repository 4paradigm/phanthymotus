"""PyTorch-free VITS2 inference using TensorRT and NumPy."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from .trt_cuda_session import CudaRuntime, TensorRTCudaSession


APP_DIR = Path(__file__).resolve().parent.parent


def _intersperse(values, item=0):
    result = [item] * (len(values) * 2 + 1)
    result[1::2] = values
    return result


def _sequence_mask(lengths, max_length=None):
    lengths = np.asarray(lengths)
    if max_length is None:
        max_length = int(lengths.max())
    return np.arange(max_length)[None, :] < lengths[:, None]


def _generate_path(duration, mask):
    batch, _, target_length, source_length = mask.shape
    cumulative = np.cumsum(duration, axis=-1).reshape(batch * source_length)
    path = _sequence_mask(cumulative, target_length).astype(mask.dtype)
    path = path.reshape(batch, source_length, target_length)
    previous = np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    path = path - previous
    return path[:, None].transpose(0, 1, 3, 2) * mask


def _istft(spectrum, n_fft: int, hop_length: int):
    frames = np.fft.irfft(spectrum, n=n_fft, axis=1).astype(np.float32)
    frame_count = frames.shape[-1]
    output_length = n_fft + hop_length * (frame_count - 1)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(
        np.float32
    )
    audio = np.zeros((frames.shape[0], output_length), dtype=np.float32)
    envelope = np.zeros(output_length, dtype=np.float32)
    for frame_index in range(frame_count):
        start = frame_index * hop_length
        audio[:, start : start + n_fft] += frames[:, :, frame_index] * window
        envelope[start : start + n_fft] += window * window
    valid = envelope > 1e-11
    audio[:, valid] /= envelope[valid]
    trim = n_fft // 2
    return audio[:, trim:-trim] if trim else audio


def _soft_clip_and_pcm(audio, sample_rate):
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.95, neginf=-0.95)
    peak = float(np.max(np.abs(audio), initial=0.0))
    if peak > 1.0:
        audio = audio / peak
    fade_samples = min(int(sample_rate * 0.020), audio.shape[-1])
    if fade_samples:
        audio[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    return np.clip(audio * 32767.0, -32768, 32767).astype(np.int16).tobytes()


class TensorRTNumpyTTSEngine:
    def __init__(self, config_path: str, engine_dir: str | None = None, replica_id: int = 0):
        self.engine_dir = Path(
            engine_dir or os.getenv("MIX_VITS_TRT_ENGINE_DIR", APP_DIR / "engines")
        )
        manifest_path = self.engine_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"TensorRT manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.sr = int(config["data"]["sampling_rate"])
        self.add_blank = bool(config["data"].get("add_blank", True))
        self.n_fft = int(config["model"].get("gen_istft_n_fft", 16))
        self.istft_hop = int(config["model"].get("gen_istft_hop_size", 4))
        self.max_text_tokens = int(os.getenv("MIX_VITS_MAX_TEXT_TOKENS", "512"))
        self.max_frames = int(os.getenv("MIX_VITS_MAX_FRAMES", "2048"))
        self.replica_id = replica_id
        self.cuda = CudaRuntime()
        self._validate_manifest()
        engines = self.manifest["engines"]
        self.encoder = self._load_engine("encoder_duration", engines)
        self.flow = self._load_engine("flow", engines)
        self.decoder = self._load_engine("decoder", engines)
        self.runtime_info = {
            "backend": "tensorrt_cuda_numpy",
            "engine_dir": str(self.engine_dir),
            "total_engine_bytes": self.manifest["total_engine_bytes"],
            "tensorrt_version": self.manifest["trtexec_version"],
        }

    def _load_engine(self, name, engines):
        entry = engines.get(name)
        if entry is None:
            raise RuntimeError(f"TensorRT manifest is missing engine {name!r}")
        return TensorRTCudaSession(
            self.engine_dir / entry["file"], self.cuda, entry["sha256"]
        )

    def _validate_manifest(self):
        import tensorrt as trt

        if int(self.manifest.get("total_engine_bytes", 0)) > 50 * 1024 * 1024:
            raise RuntimeError("TensorRT bundle exceeds the 50 MiB model budget")
        expected_major = str(self.manifest.get("tensorrt_major", ""))
        actual_major = trt.__version__.split(".", 1)[0]
        if expected_major != actual_major:
            raise RuntimeError(
                f"TensorRT major mismatch: engine={expected_major}, runtime={actual_major}"
            )
        capability = str(self.manifest.get("compute_capability", "unknown"))
        actual_capability = self.cuda.compute_capability()
        if capability != "unknown" and capability != actual_capability:
            raise RuntimeError(
                f"GPU compute capability mismatch: engine={capability}, runtime={actual_capability}"
            )

    def _get_text_ids(self, text, *, normalized=False):
        from ..frontend import cleaned_text_to_sequence_mix
        from ..frontend.cleaner import clean_text_mix, g2p_normalized_text_mix

        if normalized:
            phones, tones, langs, _ = g2p_normalized_text_mix(text)
        else:
            _, phones, tones, langs, _ = clean_text_mix(text)
        ids = cleaned_text_to_sequence_mix(phones, tones, langs)
        if self.add_blank:
            ids = tuple(_intersperse(values) for values in ids)
        return tuple(tuple(values) for values in ids)

    def _infer_ids(self, text_ids, noise_scale=0.667, length_scale=1.0):
        phone_ids, tone_ids, lang_ids = text_ids
        text_length = len(phone_ids)
        if text_length > self.max_text_tokens:
            raise ValueError(
                f"Text has {text_length} tokens; TensorRT profile limit is {self.max_text_tokens}"
            )
        outputs = self.encoder.run(
            {
                "x": np.asarray([phone_ids], dtype=np.int32),
                "x_lengths": np.asarray([text_length], dtype=np.int32),
                "tone": np.asarray([tone_ids], dtype=np.int32),
                "language": np.asarray([lang_ids], dtype=np.int32),
                "sid": np.zeros(1, dtype=np.int32),
            }
        )
        m_p = outputs["m_p"].astype(np.float32)
        logs_p = outputs["logs_p"].astype(np.float32)
        x_mask = outputs["x_mask"].astype(np.float32)
        logw = outputs["logw"].astype(np.float32)
        g = outputs["g"].astype(np.float32)
        duration = np.ceil(np.exp(logw) * x_mask * length_scale)
        y_lengths = np.maximum(np.sum(duration, axis=(1, 2)), 1).astype(np.int64)
        frame_count = int(y_lengths.max())
        if frame_count > self.max_frames:
            raise ValueError(
                f"Audio requires {frame_count} frames; TensorRT profile limit is {self.max_frames}"
            )
        y_mask = _sequence_mask(y_lengths).astype(x_mask.dtype)[:, None, :]
        attention = _generate_path(
            duration, x_mask[:, :, None, :] * y_mask[:, :, :, None]
        )
        m_p = np.matmul(attention[:, 0], m_p.transpose(0, 2, 1)).transpose(0, 2, 1)
        logs_p = np.matmul(
            attention[:, 0], logs_p.transpose(0, 2, 1)
        ).transpose(0, 2, 1)
        noise = np.random.standard_normal(m_p.shape).astype(np.float32)
        z_p = m_p + noise * np.exp(logs_p) * noise_scale
        z = self.flow.run({"z_p": z_p, "y_mask": y_mask, "g": g})["z"]
        logits = self.decoder.run({"z": z * y_mask, "g": g})[
            "decoder_logits"
        ].astype(np.float32)
        split = self.n_fft // 2 + 1
        magnitude = np.exp(logits[:, :split])
        phase = math.pi * np.sin(logits[:, split:])
        spectrum = magnitude * np.exp(1j * phase)
        audio = _istft(spectrum, self.n_fft, self.istft_hop)[0]
        return _soft_clip_and_pcm(audio, self.sr)

    def synthesize(self, text, text_ids=None, noise_scale=0.667, length_scale=1.0, **_kwargs):
        return self._infer_ids(
            text_ids or self._get_text_ids(text), noise_scale, length_scale
        )

    def synthesize_batch(self, batch_text_ids, **kwargs):
        return [self._infer_ids(text_ids, **kwargs) for text_ids in batch_text_ids]
