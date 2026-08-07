"""Checkpoint-free VITS2 inference using ONNX Runtime on CPU."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def _intersperse(values, item=0):
    result = [item] * (len(values) * 2 + 1)
    result[1::2] = values
    return result


def _sequence_mask(lengths, max_length=None):
    if max_length is None:
        max_length = int(lengths.max().item())
    return torch.arange(max_length).unsqueeze(0) < lengths.unsqueeze(1)


def _generate_path(duration, mask):
    batch, _, target_length, source_length = mask.shape
    cumulative = torch.cumsum(duration, -1).reshape(batch * source_length)
    path = _sequence_mask(cumulative, target_length).to(mask.dtype)
    path = path.reshape(batch, source_length, target_length)
    path = path - torch.nn.functional.pad(path, (0, 0, 1, 0))[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask


def _pcm16(audio, sample_rate):
    audio = torch.nan_to_num(audio, nan=0.0, posinf=0.95, neginf=-0.95)
    peak = float(audio.abs().amax().item())
    if peak > 1.0:
        audio = audio / peak
    fade_samples = min(int(sample_rate * 0.020), audio.shape[-1])
    if fade_samples:
        ramp = torch.linspace(0.0, 1.0, fade_samples, dtype=audio.dtype)
        audio[:fade_samples] *= ramp
    return (
        audio.float()
        .mul(32767.0)
        .clamp_(-32768, 32767)
        .to(torch.int16)
        .numpy()
        .tobytes()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OnnxCpuEngine:
    def __init__(self, config_path: Path, model_dir: Path, num_threads: int = 6):
        self.model_dir = Path(model_dir)
        manifest_path = self.model_dir / "onnx_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"ONNX manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))

        self.sample_rate = int(config["data"]["sampling_rate"])
        self.add_blank = bool(config["data"].get("add_blank", True))
        self.n_fft = int(config["model"].get("gen_istft_n_fft", 16))
        self.istft_hop = int(config["model"].get("gen_istft_hop_size", 4))
        self.max_text_tokens = 512
        self.max_frames = 2048
        self.num_threads = max(1, int(num_threads))

        self._validate_models()
        self.encoder = self._load("encoder_duration")
        self.flow = self._load("flow")
        self.decoder = self._load("decoder")
        self.window = torch.hann_window(self.n_fft, periodic=True)

    def _validate_models(self):
        models = self.manifest.get("models", {})
        expected = {"encoder_duration", "flow", "decoder"}
        if set(models) != expected:
            raise RuntimeError(f"ONNX manifest models must be {sorted(expected)}")
        total = 0
        for entry in models.values():
            path = self.model_dir / entry["file"]
            if not path.is_file():
                raise FileNotFoundError(path)
            size = path.stat().st_size
            total += size
            if size != int(entry["bytes"]):
                raise RuntimeError(f"ONNX size mismatch: {path}")
            if _sha256(path) != entry["sha256"]:
                raise RuntimeError(f"ONNX checksum mismatch: {path}")
        if total > 50 * 1024 * 1024:
            raise RuntimeError(f"ONNX bundle exceeds 50 MiB: {total} bytes")

    def _load(self, name):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = self.num_threads
        path = self.model_dir / self.manifest["models"][name]["file"]
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError(f"Unexpected ONNX providers: {session.get_providers()}")
        return session

    @staticmethod
    def _run(session, inputs):
        names = [output.name for output in session.get_outputs()]
        return dict(zip(names, session.run(names, inputs)))

    def _text_ids(self, text):
        from .frontend import cleaned_text_to_sequence_mix
        from .frontend.cleaner import clean_text_mix

        _, phones, tones, langs, _ = clean_text_mix(text)
        ids = cleaned_text_to_sequence_mix(phones, tones, langs)
        if self.add_blank:
            ids = tuple(_intersperse(values) for values in ids)
        return tuple(tuple(values) for values in ids)

    def text_token_count(self, text):
        """Return the encoded phone-token count used by the ONNX encoder."""
        return len(self._text_ids(text)[0])

    @torch.inference_mode()
    def synthesize(self, text, noise_scale=0.667, length_scale=1.0):
        phone_ids, tone_ids, lang_ids = self._text_ids(text)
        text_length = len(phone_ids)
        if text_length > self.max_text_tokens:
            raise ValueError(
                f"Text has {text_length} tokens; limit is {self.max_text_tokens}"
            )

        outputs = self._run(
            self.encoder,
            {
                "x": np.asarray([phone_ids], dtype=np.int32),
                "x_lengths": np.asarray([text_length], dtype=np.int32),
                "tone": np.asarray([tone_ids], dtype=np.int32),
                "language": np.asarray([lang_ids], dtype=np.int32),
                "sid": np.zeros(1, dtype=np.int32),
            },
        )
        m_p = torch.from_numpy(outputs["m_p"])
        logs_p = torch.from_numpy(outputs["logs_p"])
        x_mask = torch.from_numpy(outputs["x_mask"])
        logw = torch.from_numpy(outputs["logw"])
        g = torch.from_numpy(outputs["g"])

        duration = torch.ceil(torch.exp(logw) * x_mask * length_scale)
        y_lengths = torch.clamp_min(torch.sum(duration, (1, 2)), 1).long()
        frame_count = int(y_lengths.max().item())
        if frame_count > self.max_frames:
            raise ValueError(
                f"Audio requires {frame_count} frames; limit is {self.max_frames}"
            )
        y_mask = _sequence_mask(y_lengths).unsqueeze(1).to(x_mask.dtype)
        attention = _generate_path(
            duration, x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
        )
        m_p = torch.matmul(
            attention.squeeze(1), m_p.transpose(1, 2)
        ).transpose(1, 2)
        logs_p = torch.matmul(
            attention.squeeze(1), logs_p.transpose(1, 2)
        ).transpose(1, 2)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self._run(
            self.flow,
            {"z_p": z_p.numpy(), "y_mask": y_mask.numpy(), "g": g.numpy()},
        )["z"]
        logits = torch.from_numpy(
            self._run(
                self.decoder,
                {"z": z * y_mask.numpy(), "g": g.numpy()},
            )["decoder_logits"]
        )
        split = self.n_fft // 2 + 1
        spectrum = torch.polar(
            torch.exp(logits[:, :split]),
            math.pi * torch.sin(logits[:, split:]),
        )
        audio = torch.istft(
            spectrum,
            self.n_fft,
            self.istft_hop,
            self.n_fft,
            window=self.window,
        )[0]
        return _pcm16(audio, self.sample_rate)
