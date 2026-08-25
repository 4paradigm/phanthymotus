"""
utils/onnx_provider.py — Validate a device choice and pick the weights for it.

The plugin config exposes `device: cpu | gpu`; `ASR_MODELS` in plugins/asr.py
declares, per model, which weights each device loads. This module does not decide
*what* to run — the registry does — it decides whether a (device, weights) pair is
allowed to run, and helps adapters find the right file.

Everything here encodes a measured failure mode. All numbers are from orin5
(Orin NX, JetPack 5.11, CUDA 11.4, 6 cores) with sherpa-onnx 1.13.6+cuda, on an
idle box (services stopped, idle GR3D verified 0%), swept across 1/2/4
`num_threads`, reporting the steady-state median:

| model                                 | dtype | CPU t=2 | CUDA t=2 | ratio  |
|---------------------------------------|-------|---------|----------|--------|
| streaming paraformer (30.7 s audio)   | int8  | 3295 ms |  8394 ms |  0.39x |
| streaming paraformer                  | fp32  | 8702 ms |  1859 ms |  4.68x |
| streaming paraformer                  | fp16  |42890 ms |  2077 ms | 20.65x |
| X-ASR beam search (28.7 s audio)      | int8  | 2645 ms |  3294 ms |  0.80x |
| offline SenseVoice (28.7 s audio)     | int8  | 1996 ms |  2753 ms |  0.73x |
| offline SenseVoice                    | fp32  | 4792 ms |   416 ms | 11.52x |
| offline SenseVoice                    | fp16  | 8117 ms |   344 ms | 23.60x |
| Matcha TTS (13.3 s audio)             | fp32  | 1784 ms |   416 ms |  4.28x |

Three things follow, and they are the three rules below:

1. **int8 loses on the GPU, always.** ONNX Runtime's CUDA provider has no kernels
   for quantised ops, so it partitions the graph and falls back to CPU node by
   node, adding a host<->device copy at every boundary. Confirmed two ways: fp32
   and fp16 on CUDA are completely insensitive to CPU thread count (417/416/415
   and 343/344/346 ms) with GR3D pinned at 95-97%, while int8 on CUDA scales with
   CPU threads (3793 -> 2753 -> 1905 for SenseVoice). int8 on CUDA is also numerically noisier —
   requantising at every partition boundary changed 3 of 4 SenseVoice transcripts
   versus the same model on CPU, including `不然` -> `主然`.

2. **fp16 loses on the CPU, catastrophically.** ONNX Runtime has no fp16 CPU
   kernels and casts everything: 42890 ms where int8 took 3295 ms.

3. **Neither speed nor self-consistency proves a pair works.** Streaming
   paraformer fp16 on CUDA was fast (2077 ms), identical across all three thread
   counts, and created a session without complaint — while emitting nothing but
   `</s>`. The same fp16 file on CPU transcribed correctly, so the conversion was
   fine and the CUDA+fp16+streaming combination is not. That is why every
   (model, device) entry in ASR_MODELS must have had its *output read* before
   being listed, and why this module can only catch the mechanical mistakes below,
   not that one.

`int16` is not a middle ground worth trying: ONNX Runtime's int16 quantisation is
newer than int8, the CUDA provider has no kernels for it either, and the CPU side
lacks the dot-product paths that make int8 fast.

**gpu costs memory, and most of it is unreturnable.** Measured on the same box with
only ASR resident, SenseVoice:

    cpu (int8)  build +542 MB, back to  129 MB after the adapter is dropped
    gpu (fp16)  build +1968 MB, still  1516 MB after the adapter is dropped

The residue is the CUDA context and its memory pool: once a process has touched
CUDA it does not give that back. On a 7.4 GB Orin already running vop (YOLO), OCR
(TensorRT) and TTS, enabling gpu ASR was enough to exhaust memory and get
perception restarted in a loop. So `cpu` stays the default even on images that
*can* use the GPU — gpu is a deliberate choice for a box with headroom.

**And the first gpu inference is the expensive one.** 1659 ms against a 58 ms
steady state: lazy kernel loading, cuDNN autotuning and the memory pool all land
on it. plugins/asr.py warms the adapter up with one second of silence after
building so an operator never pays that on a real utterance.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

log = logging.getLogger(__name__)

VALID_DEVICES = ("cpu", "gpu")


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """True when the installed sherpa_onnx wheel bundles the CUDA provider.

    The marker is `sherpa_onnx/lib/libonnxruntime_providers_cuda.so`, which only a
    `-DSHERPA_ONNX_ENABLE_GPU=ON` build ships. jetson jp5.11 installs such a wheel
    from COS; jp6.1 is CUDA 12.6 and cannot use it, and x86 dev hosts get the PyPI
    CPU wheel — see Dockerfile.jetson. This deliberately does not probe the driver
    or create a session: on Jetson the wheel is built for the same L4T release it
    is deployed on, and a probe would cost a full ONNX Runtime session at import.
    """
    try:
        import sherpa_onnx
    except ImportError:
        return False
    lib = Path(sherpa_onnx.__file__).resolve().parent / "lib"
    return (lib / "libonnxruntime_providers_cuda.so").is_file()


def is_quantised(model_paths: Iterable[str]) -> bool:
    """True when any of these weight files is int8.

    *Any*, not all: X-ASR ships an int8 encoder and joiner beside an fp32 decoder,
    and CUDA still lost on it, so one quantised file disqualifies the set.

    Filename, not graph inspection: `.int8.onnx` is the naming every sherpa-onnx
    bundle uses, and it is already what asr.py and x_asr.py match on to *prefer*
    the quantised file. Parsing the protobuf for QuantizeLinear would be more
    precise but costs a read of a 636 MB file at startup.
    """
    return any("int8" in Path(p).name.lower() for p in model_paths if p)


def is_fp16(model_paths: Iterable[str]) -> bool:
    """True when any of these weight files is fp16, by the same naming convention."""
    return any("fp16" in Path(p).name.lower() for p in model_paths if p)


def normalize_device(value: str | None, legacy_hw_provider: str | None = None) -> str:
    """Coerce a configured device to `cpu`/`gpu`.

    `legacy_hw_provider` accepts the pre-`device` config key so a deployment that
    mounts its own config.yaml keeps working; `cuda` there means `gpu`.
    """
    raw = (value or "").strip().lower()
    if not raw and legacy_hw_provider:
        legacy = legacy_hw_provider.strip().lower()
        raw = "gpu" if legacy in ("gpu", "cuda") else "cpu" if legacy == "cpu" else ""
        if raw:
            log.info("[onnx_provider] using legacy hw_provider=%r as device=%r",
                     legacy_hw_provider, raw)
    if not raw:
        return "cpu"
    if raw in ("cuda", "nvidia"):
        return "gpu"
    if raw not in VALID_DEVICES:
        log.warning("[onnx_provider] unknown device %r, using cpu", value)
        return "cpu"
    return raw


def provider_for_device(device: str, model_paths: Sequence[str] = ()) -> str:
    """Return the ONNX Runtime provider to use, refusing the known-bad pairs.

    Falls back rather than raising: a wrong device in a baked config.yaml should
    degrade to a working recogniser, not stop perception from booting. The
    dashboard path validates separately and reports the error there, where someone
    is watching.
    """
    device = normalize_device(device)
    paths = [p for p in model_paths if p]

    if device == "gpu":
        if not cuda_available():
            log.warning("[onnx_provider] device=gpu but the installed sherpa_onnx "
                        "wheel has no CUDA provider — falling back to cpu")
            return "cpu"
        if is_quantised(paths):
            # Registry bug: a gpu entry must not point at int8 weights.
            log.error("[onnx_provider] device=gpu selected int8 weights (%s) — "
                      "ONNX Runtime's CUDA provider falls back to CPU per node on "
                      "quantised ops and measured 1.25x-3.3x slower. Using cpu; "
                      "fix the ASR_MODELS entry.",
                      ", ".join(os.path.basename(p) for p in paths))
            return "cpu"
        return "cuda"

    if is_fp16(paths):
        # Registry bug in the other direction, and a much worse one: 42890 ms vs
        # 3295 ms for int8 on the streaming model.
        log.error("[onnx_provider] device=cpu selected fp16 weights (%s) — "
                  "ONNX Runtime has no fp16 CPU kernels and casts everything, "
                  "measured over 10x slower than int8. Fix the ASR_MODELS entry.",
                  ", ".join(os.path.basename(p) for p in paths))
    return "cpu"


def pick_weights(model_dir: str, *candidates: str) -> str:
    """Return the first candidate filename that exists in model_dir.

    Falls back to the *last* candidate when none exist, so the caller's own
    "model files not found" error names the file it actually wanted rather than
    an empty string. Adapters pass candidates in device-appropriate order: a gpu
    directory holds fp16/fp32 weights, a cpu directory holds int8.
    """
    for name in candidates:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(model_dir, candidates[-1]) if candidates else ""
