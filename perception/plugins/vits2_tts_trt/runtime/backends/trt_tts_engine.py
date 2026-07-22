"""Checkpoint-free VITS2 inference using three direct TensorRT engines."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import torch

from .trt_session import TensorRTSession

logger = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent.parent


def _intersperse(values, item=0):
    result = [item] * (len(values) * 2 + 1)
    result[1::2] = values
    return result


def _sequence_mask(lengths, max_length=None):
    max_length = int(lengths.max().item()) if max_length is None else max_length
    return torch.arange(max_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def _generate_path(duration, mask):
    batch, _, target_length, source_length = mask.shape
    cumulative = torch.cumsum(duration, -1).reshape(batch * source_length)
    path = _sequence_mask(cumulative, target_length).to(mask.dtype)
    path = path.reshape(batch, source_length, target_length)
    path = path - torch.nn.functional.pad(path, (0, 0, 1, 0))[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask


def _soft_clip_and_pcm(audio, sample_rate):
    audio = torch.nan_to_num(audio, nan=0.0, posinf=0.95, neginf=-0.95)
    peak = float(audio.abs().amax().item())
    if peak > 1.0:
        audio = audio / peak
    fade_samples = min(int(sample_rate * 0.020), audio.shape[-1])
    if fade_samples:
        ramp = torch.linspace(0.0, 1.0, fade_samples, device=audio.device, dtype=audio.dtype)
        audio[:fade_samples] *= ramp
    return audio.float().mul(32767.0).clamp_(-32768, 32767).to(torch.int16).cpu().numpy().tobytes()


class TensorRTTTSEngine:
    def __init__(self, config_path: str, engine_dir: str | None = None, replica_id: int = 0):
        self.engine_dir = Path(engine_dir or os.getenv("MIX_VITS_TRT_ENGINE_DIR", APP_DIR / "engines"))
        manifest_path = self.engine_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"TensorRT manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with Path(config_path).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.sr = int(config["data"]["sampling_rate"])
        self.add_blank = bool(config["data"].get("add_blank", True))
        self.n_fft = int(config["model"].get("gen_istft_n_fft", 16))
        self.istft_hop = int(config["model"].get("gen_istft_hop_size", 4))
        self.max_text_tokens = int(os.getenv("MIX_VITS_MAX_TEXT_TOKENS", "512"))
        self.max_frames = int(os.getenv("MIX_VITS_MAX_FRAMES", "2048"))
        self.replica_id = replica_id
        self._validate_manifest()

        engines = self.manifest["engines"]
        self.encoder = self._load_engine("encoder_duration", engines)
        self.flow = self._load_engine("flow", engines)
        self.decoder = self._load_engine("decoder", engines)
        self.window = torch.hann_window(self.n_fft, periodic=True, device="cuda")
        self.runtime_info = {
            "backend": "tensorrt",
            "engine_dir": str(self.engine_dir),
            "total_engine_bytes": self.manifest["total_engine_bytes"],
            "tensorrt_version": self.manifest["trtexec_version"],
        }

    def _load_engine(self, name, engines):
        entry = engines.get(name)
        if entry is None:
            raise RuntimeError(f"TensorRT manifest is missing engine {name!r}")
        return TensorRTSession(self.engine_dir / entry["file"], entry["sha256"])

    def _validate_manifest(self):
        import tensorrt as trt

        if int(self.manifest.get("total_engine_bytes", 0)) > 50 * 1024 * 1024:
            raise RuntimeError("TensorRT bundle exceeds the 50 MiB model budget")
        expected_major = str(self.manifest.get("tensorrt_major", ""))
        actual_major = trt.__version__.split(".", 1)[0]
        if expected_major != actual_major:
            raise RuntimeError(f"TensorRT major mismatch: engine={expected_major}, runtime={actual_major}")
        capability = str(self.manifest.get("compute_capability", "unknown"))
        actual_capability = ".".join(map(str, torch.cuda.get_device_capability()))
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

    @torch.inference_mode()
    def _infer_ids(self, text_ids, noise_scale=0.667, length_scale=1.0):
        phone_ids, tone_ids, lang_ids = text_ids
        text_length = len(phone_ids)
        if text_length > self.max_text_tokens:
            raise ValueError(f"Text has {text_length} tokens; TensorRT profile limit is {self.max_text_tokens}")
        device = "cuda"
        outputs = self.encoder.run({
            "x": torch.tensor([phone_ids], dtype=torch.int32, device=device),
            "x_lengths": torch.tensor([text_length], dtype=torch.int32, device=device),
            "tone": torch.tensor([tone_ids], dtype=torch.int32, device=device),
            "language": torch.tensor([lang_ids], dtype=torch.int32, device=device),
            "sid": torch.zeros(1, dtype=torch.int32, device=device),
        })
        m_p = outputs["m_p"].float()
        logs_p = outputs["logs_p"].float()
        x_mask = outputs["x_mask"].float()
        logw = outputs["logw"].float()
        g = outputs["g"].float()
        w_ceil = torch.ceil(torch.exp(logw) * x_mask * length_scale)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, (1, 2)), 1).long()
        frame_count = int(y_lengths.max().item())
        if frame_count > self.max_frames:
            raise ValueError(f"Audio requires {frame_count} frames; TensorRT profile limit is {self.max_frames}")
        y_mask = _sequence_mask(y_lengths).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
        attn = _generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow.run({"z_p": z_p, "y_mask": y_mask, "g": g})["z"]
        logits = self.decoder.run({"z": z * y_mask, "g": g})["decoder_logits"].float()
        split = self.n_fft // 2 + 1
        magnitude = torch.exp(logits[:, :split])
        phase = math.pi * torch.sin(logits[:, split:])
        spectrum = torch.polar(magnitude, phase)
        audio = torch.istft(
            spectrum,
            self.n_fft,
            self.istft_hop,
            self.n_fft,
            window=self.window,
        )[0]
        return _soft_clip_and_pcm(audio, self.sr)

    def synthesize(self, text, text_ids=None, noise_scale=0.667, length_scale=1.0, **_kwargs):
        return self._infer_ids(text_ids or self._get_text_ids(text), noise_scale, length_scale)

    def synthesize_batch(self, batch_text_ids, **kwargs):
        return [self._infer_ids(text_ids, **kwargs) for text_ids in batch_text_ids]
