#!/usr/bin/env python3
"""
plugins/speaker_runtime.py — 声纹 embedding 推理封装（无 ROS 依赖）。

Mirrors `plugins/face_runtime.py`'s role: everything that touches a model lives
here, so `plugins/speaker.py` only deals with ROS, MCP and lifecycle.

One model, and it runs on **sherpa-onnx's own ONNX Runtime** — the same one ASR
and TTS already use. That is the whole reason this plugin is ~300 lines where
face needed a separate process: there is no second ONNX Runtime in the address
space, so none of `plugins/ort_worker.py`'s machinery applies (see
`plugins/face_runtime.py` lines 40-66 for what that machinery is avoiding —
an exception on jp6.1 and a SIGSEGV that kills all of perception on jp5.11).
Do not switch this to the standalone `onnxruntime` package.

## Input length is quantised, and that is not an optimisation

Measured 2026-10-09 in the perception image on both Orins: with a **fixed** input
shape, 95 consecutive calls grow RSS by **0.0 MB**. Feed it real VAD segments
instead — every one a different number of samples, so every call a new shape —
and jp5.11 was still climbing +60 MB after 100 segments with no plateau in sight.
What grows is ONNX Runtime's CPU arena, which allocates per input shape; sherpa
exposes no way to disable it, so the only lever is the input side.

So every waveform is rounded up to a `GRID_S` boundary with zeros and truncated
at `MAX_S`. That collapsed 102 distinct shapes to 4 in the measurement, and it
has two more effects that matter as much:

* **Latency becomes constant** — p50 87-99 ms instead of 35 ms at 1 s rising to
  380 ms (p90 830 ms) past 6 s.
* **It keeps the embedding cheaper than ASR**, which is what lets the two run
  concurrently for free. `plugins/asr.py` runs this on a thread beside
  `transcribe()`; the overlap is only free while this is the shorter of the two.

Zero-padding is not free but it is *systematic*: cosine against the unpadded
embedding was p50 0.966 / p05 0.86, and enrolment pads the same way recognition
does, so the offset largely cancels. **Truncation is a different matter** — see
`SpeakerEmbedder.embed_windowed`.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from utils.model_downloader import ensure_speaker_model
from utils.onnx_provider import normalize_device

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

# Rounding grid and ceiling for the input waveform, in seconds. 0.5 s × 6 s = 12
# reachable shapes, which is what bounds the arena. Raising MAX_S costs shapes
# and latency; lowering it throws away speech.
GRID_S = 0.5
MAX_S = 6.0

# `model: off` 不是一个模型，是「完全不加载」。选它的时候 extractor 不会被构造，
# 权重不进内存（cpu 实测 +84 MB，gpu +478 MB），也不占 CPU —— 和「卡片 stop」不同，
# 后者只是停止归因、引擎还留在内存里等着重新启用。
#
# 放在 `model` 的枚举里而不是单独一个 enabled 开关：一个控件三种状态（关闭 / 这个
# 模型 / 以后的模型），没有「关着但选了模型」这种说不清的组合。
MODEL_OFF = "off"

DEFAULT_SPEAKER_MODEL = "campplus_zh_en"
DEFAULT_SPEAKER_MODEL_DIR = "/models/speaker/campplus_zh_en"

# Which embedding network to run. One entry today; the registry exists so that
# adding a second is a table row rather than a hunt through the file, and so the
# card can offer the choice.
#
# What a new entry has to supply: a 16 kHz model (this project's audio bus is
# `audio/pcm-16k`, and a mismatch would need a resampler that does not exist
# here), and its own pinned bytes in `utils/model_downloader.SPEAKER_MODEL_BUNDLES`.
# `dim` is **not** listed: it is read from `extractor.dim` at load time, because
# a hardcoded dimension that disagrees with the weights is exactly the failure
# `IdentityDB` now refuses to let through.
#
# Changing the model invalidates the database — embeddings from different networks
# are not comparable. `IdentityDB` records the model name and refuses to open a
# database written by a different one, so this is a deploy-time choice, not
# something to flip on a running robot with a populated roster.
SPEAKER_MODELS = {
    # 3D-Speaker CAM++ trained on 200k speakers, Chinese **and** English.
    # 28.3 MB, 192-d. Apache-2.0.
    #
    # The bilingual variant rather than `campplus_sv_zh-cn_16k-common`: ASR here
    # already runs three models covering zh and en, so a Chinese-only voiceprint
    # would degrade silently the moment somebody speaks English — and the two
    # files are the same size and the same dimension, so there is no trade.
    "campplus_zh_en": {
        "dir": DEFAULT_SPEAKER_MODEL_DIR,
        "weights": "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        "bundle": "campplus_zh_en",
        "description": "3D-Speaker CAM++ (200k speakers, 中英双语, 192-d, 28 MB)",
    },
}


def is_off(model: str) -> bool:
    """Is this the sentinel that means "do not load anything"?"""
    return str(model or "").strip().lower() in (MODEL_OFF, "none", "disabled", "")


def speaker_bundle_for(model: str, model_dir: str) -> tuple[str, str]:
    """Resolve (bundle name, directory) for a model, for the download step.

    Mirrors `face_runtime.face_bundle_for`: the plugin needs to fetch weights
    before it can construct anything, and it should not have to know that a
    model name and a download bundle are two different registries.
    """
    spec = SPEAKER_MODELS.get(model)
    if spec is None:
        raise ValueError(
            f"unknown speaker model {model!r}; this build has "
            f"{sorted(SPEAKER_MODELS)}"
        )
    return spec["bundle"], (model_dir or spec["dir"])


def grid_samples(count: int) -> int:
    """Round a sample count up to the padding grid, capped at MAX_S.

    Pure, so the shape arithmetic is testable without a model.
    """
    grid = int(SAMPLE_RATE * GRID_S)
    cap = int(SAMPLE_RATE * MAX_S)
    count = max(1, min(int(count), cap))
    return min(cap, ((count + grid - 1) // grid) * grid)


def reachable_shapes() -> int:
    """How many distinct input widths `grid_samples` can produce.

    Asserted by the tests. If a change to GRID_S/MAX_S pushes this up, the ORT
    arena grows with it — which is the thing the quantisation exists to stop.
    """
    return int(MAX_S / GRID_S)


class SpeakerEmbedderError(RuntimeError):
    """The model is present but cannot be used as-is."""


class SpeakerEmbedder:
    """One sherpa-onnx speaker-embedding extractor, plus the input conditioning.

    Thread safety: `compute` is called from the ASR worker thread (on a thread of
    its own, beside `transcribe`) and from arbitrary `ThreadingHTTPServer`
    threads serving `register_*`. It takes a lock.

    `plugins/asr.py` shares one `ASRAdapter` across instances with **no** lock
    today. That is either safe or an existing latent bug; this does not copy the
    uncertainty — at utterance rate the contention is unmeasurable.
    """

    def __init__(
        self,
        model: str = DEFAULT_SPEAKER_MODEL,
        model_dir: str = "",
        device: str = "cpu",
        num_threads: int = 2,
        warmup: bool = True,
        on_status=None,
    ):
        import sherpa_onnx

        from utils.onnx_provider import provider_for_device

        spec = SPEAKER_MODELS.get(model)
        if spec is None:
            raise SpeakerEmbedderError(
                f"unknown speaker model {model!r}; this build has "
                f"{sorted(SPEAKER_MODELS)}. Adding one is a row in SPEAKER_MODELS, "
                f"but read the note there about 16 kHz and pinned bytes first."
            )
        bundle, directory = speaker_bundle_for(model, model_dir)

        def status(text: str) -> None:
            if on_status:
                on_status(text)

        status(f"下载声纹模型 {model}")
        paths = ensure_speaker_model(directory, bundle=bundle)
        weights = paths[spec["weights"]]

        # `device` defaults to cpu throughout this plugin, and the reason is not
        # the memory cost (a CUDA context for this model measured +478 MB, not the
        # ~1.4 GB config.yaml quotes for ASR). It is that the 1.62x speedup does
        # not survive the pipeline: with `asr=cpu` this already hides entirely
        # behind ASR's 230 ms, and with `asr=gpu` the two CUDA sessions serialise
        # on the one GPU, so moving this there buys 3 ms. Measured on Orin 6.
        device = normalize_device(device) or "cpu"
        provider = provider_for_device(device, (weights,))
        self._device = "gpu" if provider != "cpu" else "cpu"
        self._provider = provider
        self._model = model
        self._weights = weights

        status(f"加载声纹模型 {model}")
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=weights, num_threads=max(1, int(num_threads)),
            debug=False, provider=provider,
        )
        if not config.validate():
            raise SpeakerEmbedderError(
                f"sherpa-onnx rejected the speaker extractor config for {weights}"
            )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self._dim = int(self._extractor.dim)
        self._lock = threading.RLock()

        if warmup:
            # Load measured 0.92 s (jp6.1) / 1.65 s (jp5.11), and the first
            # inference carries the lazy-kernel cost. Pay it here rather than on
            # the operator's first utterance — same reasoning as the ASR adapter.
            self.embed(np.zeros(int(SAMPLE_RATE * MAX_S), dtype=np.float32))
        log.info("[speaker] extractor loaded: model=%s dim=%d device=%s provider=%s "
                 "weights=%s", model, self._dim, self._device, provider, weights)

    # ── properties ────────────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def device(self) -> str:
        return self._device

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    # ── input conditioning ────────────────────────────────────────────────

    @staticmethod
    def pcm16_to_float(pcm: bytes) -> np.ndarray:
        """int16 LE bytes → float32 in [-1, 1).

        `np.frombuffer` rather than `struct.unpack` plus a list comprehension:
        the ASR adapter does the latter over 48 000 samples per utterance, which
        is pure-Python work on the critical path. Not changing that here, but not
        copying it either.
        """
        if len(pcm) % 2:
            pcm = pcm[:-1]
        return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0

    @staticmethod
    def condition(samples: np.ndarray) -> np.ndarray:
        """Truncate to MAX_S and zero-pad up to the grid. See the module docstring."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        width = grid_samples(len(samples))
        if len(samples) >= width:
            return np.ascontiguousarray(samples[:width])
        return np.concatenate(
            [samples, np.zeros(width - len(samples), dtype=np.float32)]
        )

    # ── inference ─────────────────────────────────────────────────────────

    def embed(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """One L2-normalised embedding for one waveform.

        Normalised here rather than at the call site so a dot product against
        `IdentityDB`'s matrix *is* the cosine. sherpa does not promise unit norm.
        """
        conditioned = self.condition(samples)
        with self._lock:
            stream = self._extractor.create_stream()
            stream.accept_waveform(sample_rate=sample_rate, waveform=conditioned)
            stream.input_finished()
            vector = np.array(self._extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            raise SpeakerEmbedderError("extractor returned a zero-norm embedding")
        return vector / norm

    def embed_windowed(
        self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE,
        split_above_s: float = 4.0,
    ) -> tuple[np.ndarray, float | None]:
        """Embedding plus a one-number verdict on whether the clip holds one voice.

        Returns `(embedding, coherence)`. `coherence` is `None` for a clip short
        enough to trust as single-speaker, otherwise the cosine between the
        **leading** and **trailing** windows — and a low value means the two
        describe different people, i.e. the clip contains a speaker change.

        Leading-vs-trailing rather than leading-vs-whole-clip, and that is a
        correction: the first version compared the 4 s lead against the whole
        clip, but `condition()` truncates at `MAX_S`, so on a 29 s segment
        measured on Orin 5 it was really comparing *the first 4 s against the
        first 6 s* — both from the same end. Coherence came back 0.954 on clips
        that had had 23 unexamined seconds. A change after the sixth second was
        invisible. Comparing the two ends costs the same two embeddings and
        covers the clip however long it is.

        Clips between `split_above_s` and `2 * split_above_s` have overlapping
        windows, so a change in their overlap is still softened. That is the
        floor of what two embeddings can see; widening it means more of them.

        This is not a theoretical worry. Truncation was measured against 73 real
        VAD segments longer than 3.5 s on Orin 5: cosine(full, first 3 s) had
        p50 0.843 but **p05 0.197, min 0.157**. A window and its own parent clip
        cannot be unrelated unless the parent is a blend of two voices — so that
        tail *is* the multi-speaker rate in captured office audio, and it is why
        a fixed window cannot simply replace the full clip.

        Two embeddings, not a sliding window of N. A sliding window would localise
        *where* the change is; this only answers *whether* there is one, which is
        all the caller does anything with — and it answers it for a fixed cost
        instead of one proportional to the clip.

        The returned embedding is the **leading window's**. A VAD segment opens
        on whoever started talking, so the lead is the part most likely to be one
        person; if the clip does hold one voice both ends agree anyway, and if it
        does not, the lead is the better of the two guesses. The caller decides
        what to do with a low `coherence` — `plugins/speaker.py` reports
        `multi_speaker` and withholds the identity rather than guessing.
        """
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        rate = int(sample_rate or SAMPLE_RATE)
        duration = len(samples) / float(rate)
        if duration <= split_above_s:
            return self.embed(samples, sample_rate), None
        window = int(rate * min(split_above_s, MAX_S))
        lead = self.embed(samples[:window], sample_rate)
        tail = self.embed(samples[-window:], sample_rate)
        return lead, float(np.dot(lead, tail))

    def close(self) -> None:
        with self._lock:
            self._extractor = None


__all__ = [
    "DEFAULT_SPEAKER_MODEL",
    "DEFAULT_SPEAKER_MODEL_DIR",
    "GRID_S",
    "MAX_S",
    "SAMPLE_RATE",
    "MODEL_OFF",
    "SPEAKER_MODELS",
    "SpeakerEmbedder",
    "SpeakerEmbedderError",
    "grid_samples",
    "is_off",
    "reachable_shapes",
    "speaker_bundle_for",
]
