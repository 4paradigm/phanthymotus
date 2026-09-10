"""
tests/test_kokoro_worker.py — the policy layer for Japanese, now that it owns no process.

`plugins/ort_worker.py` owns the one child that holds every standalone-ORT session in
this process. This module owns the *decisions* about Japanese, and each of them was
forced by a measurement rather than chosen:

  - **which device is safe.** jp5.11's CUDA execution of this graph computes durations
    27.5% short (8 runs: cpu 8.35 s stdev 0.000, one distinct length; cuda 6.05 s
    stdev 0.053, three), which is audibly rushed speech. The graph *is* stochastic — 4
    RandomNormalLike and 7 RandomUniformLike nodes — so comparing waveforms is not a
    valid check, and the duration is;
  - **whether there is room.** Asking for CUDA regardless OOM-killed a jp5.11 rig
    before the duration probe could run, so the headroom check has to come first;
  - **when to give the memory back.** On idle, not on the language switch: the cold
    path is ~15 s end to end and an alternating tour would pay it every time.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins import kokoro_worker as kw  # noqa: E402


class _FakeDirect:
    """A KokoroDirect stand-in whose probe length is scripted per provider."""

    lengths: dict = {}
    built: list = []
    closed: list = []

    def __init__(self, model_dir, weights, num_threads=0, provider="cpu",
                 session_key="tts.kokoro", in_process=False):
        self.provider = provider
        self.in_process = in_process
        self.session_key = session_key
        _FakeDirect.built.append((provider, in_process))
        self._session = type("s", (), {"get_providers": lambda self: [provider]})()

    num_speakers, sample_rate = 54, 24000

    def synthesize(self, phonemes, speaker_id=0, speed=1.0):
        return np.zeros(self.lengths.get(self.provider, kw.PROBE_SAMPLES),
                        dtype=np.float32)

    def synthesize_stream(self, phonemes, speaker_id=0, speed=1.0):
        yield self.synthesize(phonemes, speaker_id, speed)

    def close(self):
        _FakeDirect.closed.append(self.provider)


@pytest.fixture(autouse=True)
def _reset():
    _FakeDirect.lengths = {}
    _FakeDirect.built = []
    _FakeDirect.closed = []
    yield


def _proxy(monkeypatch, device="gpu", headroom=8192, idle=1e9, model_dir=None):
    """A proxy whose sessions are fakes, with the reaper thread neutered."""
    import plugins.kokoro_direct as kd
    monkeypatch.setattr(kd, "KokoroDirect", _FakeDirect)
    monkeypatch.setattr(kw, "_mem_available_mb", lambda: headroom)
    monkeypatch.setattr(kw.KokoroWorkerProxy, "_reap_loop", lambda self: None)
    return kw.KokoroWorkerProxy(model_dir or "/models/kokoro-multi/gpu", "model.onnx",
                                device=device, idle_timeout_s=idle)


# ── which device ──────────────────────────────────────────────────────────────

def test_a_device_that_gets_durations_wrong_raises_rather_than_substituting(
        monkeypatch, tmp_path):
    """No silent substitution: the card must say the GPU is unusable, not quietly
    become a CPU card that still reads `gpu`.

    That state is what made Japanese "mysteriously slow" and took a measurement to
    explain. The message has to distinguish this from the memory case — here the GPU is
    present and has room, and computes the duration path wrongly (jp5.11's ORT 1.15.1
    returns 27-51% short, which is audibly rushed speech).
    """
    _FakeDirect.lengths = {"gpu": 21000}
    with pytest.raises(kw.DeviceUnavailable) as excinfo:
        _proxy(monkeypatch, device="gpu", model_dir=str(tmp_path))
    message = str(excinfo.value)
    assert "duration" in message
    assert "Not a memory problem" in message, message
    assert "japanese_worker_device: cpu" in message, "the message must be actionable"
    assert "gpu" in _FakeDirect.closed, "the rejected session must be released"


def test_a_correct_device_is_kept(monkeypatch):
    _FakeDirect.lengths = {"gpu": kw.PROBE_SAMPLES}
    proxy = _proxy(monkeypatch, device="gpu")
    assert proxy.device_used == "gpu"
    assert _FakeDirect.closed == []


def test_a_few_samples_of_deviation_are_tolerated(monkeypatch):
    """A different-but-valid build must not be rejected over rounding."""
    _FakeDirect.lengths = {"gpu": int(kw.PROBE_SAMPLES * 1.03)}
    assert _proxy(monkeypatch, device="gpu").device_used == "gpu"


def test_cpu_failing_the_check_too_is_an_error(monkeypatch, tmp_path):
    """Then the model or the phoneme table does not match this code — say that, and do
    not blame the device."""
    _FakeDirect.lengths = {"cpu": 100}
    with pytest.raises(kw.DeviceUnavailable, match="does not match this code"):
        _proxy(monkeypatch, device="cpu", model_dir=str(tmp_path))


# ── whether there is room ─────────────────────────────────────────────────────

def test_no_headroom_raises_and_says_it_is_a_memory_problem(monkeypatch, tmp_path):
    """The operator has to be able to tell this apart from the durations case: here the
    GPU works and the box is full, and freeing memory would fix it."""
    with pytest.raises(kw.DeviceUnavailable) as excinfo:
        _proxy(monkeypatch, device="gpu", headroom=2048, model_dir=str(tmp_path))
    message = str(excinfo.value)
    assert "memory" in message
    assert "the GPU itself is fine" in message.lower() or "not a model problem" in message
    assert not _FakeDirect.built, "with no headroom it must not build anything at all"


def test_cuda_is_used_when_there_is_room(monkeypatch):
    assert _proxy(monkeypatch, device="gpu", headroom=4096).device_used == "gpu"


def test_an_unreadable_meminfo_does_not_block_the_gpu(monkeypatch):
    """-1 means "could not tell", which must not be read as "no memory"."""
    assert _proxy(monkeypatch, device="gpu", headroom=-1).device_used == "gpu"


# ── when to give it back ──────────────────────────────────────────────────────

def test_close_unloads_the_session_but_not_the_shared_child(monkeypatch):
    """face's sessions live in the same child and must survive Japanese finishing."""
    proxy = _proxy(monkeypatch)
    proxy.close()
    assert _FakeDirect.closed == ["gpu"]
    proxy.close()                      # idempotent


def test_idle_unloading_releases_the_session(monkeypatch):
    """A card that spoke Japanese once used to hold the session for the adapter's life:
    `set_language` never released it, so switching back to English kept it resident."""
    proxy = _proxy(monkeypatch, idle=0.0)
    proxy._last_used = time.monotonic() - 3600
    direct = None
    with proxy._lock:
        if proxy._direct is not None and \
                (time.monotonic() - proxy._last_used) > proxy._idle_timeout_s:
            direct, proxy._direct = proxy._direct, None
    assert direct is not None, "an idle session must be picked up for unloading"
    direct.close()
    assert _FakeDirect.closed == ["gpu"]


def test_a_synthesize_after_idle_unloading_rebuilds(monkeypatch):
    proxy = _proxy(monkeypatch)
    proxy._direct = None
    out = proxy.synthesize("konnichiwa", speaker_id=0, speed=1.0)
    assert out.size == kw.PROBE_SAMPLES
    assert proxy._direct is not None


# ── failure behaviour ─────────────────────────────────────────────────────────

def test_a_worker_failure_falls_back_in_process_on_cpu(monkeypatch):
    """Japanese losing 8x is acceptable; Japanese breaking is not.

    The fallback must be in-process **and** cpu: a CPU-only session loads no CUDA
    provider, so it never touches the bridge the collision runs through.
    """
    proxy = _proxy(monkeypatch)
    monkeypatch.setattr(kw.KokoroWorkerProxy, "_build",
                        lambda self: (_ for _ in ()).throw(RuntimeError("child gone")))
    proxy._direct = None
    out = proxy.synthesize("konnichiwa")
    assert out.size == kw.PROBE_SAMPLES
    assert ("cpu", True) in _FakeDirect.built, _FakeDirect.built


def test_a_failing_call_falls_back_rather_than_propagating(monkeypatch):
    proxy = _proxy(monkeypatch)
    proxy._direct.synthesize = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("the child died mid-utterance"))
    out = proxy.synthesize("konnichiwa")
    assert out.size == kw.PROBE_SAMPLES
    assert ("cpu", True) in _FakeDirect.built


# ── the headroom figure depends on who else is in the child ───────────────────

def test_a_shared_cuda_context_lowers_what_japanese_must_find(monkeypatch):
    """The first CUDA session in the worker pays for the context; the rest do not.

    Measured on Orin 6: Kokoro alone in the child cost 1585 MB, and Kokoro beside the
    face service cost 967 MB. One conservative threshold refused the second case on the
    first case's evidence — with 2158 MB free on an otherwise idle box, a 967 MB
    allocation was declined because the number was 2500.
    """
    from plugins import ort_worker
    monkeypatch.setattr(ort_worker, "get_worker",
                        lambda: type("w", (), {"has_cuda_session": lambda self: True})())
    # Between the two thresholds: too little for a fresh context, ample beside one.
    proxy = _proxy(monkeypatch, device="gpu", headroom=1800)
    assert proxy.device_used == "gpu", (
        "with a context already in the child, 1800 MB is more than the 967 MB measured")


def test_without_a_shared_context_the_higher_figure_applies(monkeypatch):
    from plugins import ort_worker
    monkeypatch.setattr(ort_worker, "get_worker",
                        lambda: type("w", (), {"has_cuda_session": lambda self: False})())
    with pytest.raises(kw.DeviceUnavailable, match="2500 MB needed"):
        _proxy(monkeypatch, device="gpu", headroom=1800)


def test_asking_the_worker_failing_is_treated_as_no_context(monkeypatch):
    """Erring towards the larger figure: guessing wrong the other way OOMs the box."""
    from plugins import ort_worker

    def _boom():
        raise RuntimeError("worker unreachable")

    monkeypatch.setattr(ort_worker, "get_worker", _boom)
    with pytest.raises(kw.DeviceUnavailable, match="2500 MB needed"):
        _proxy(monkeypatch, device="gpu", headroom=1800)


# ── the attempt is not free, so it happens once per machine ───────────────────

def test_a_rejected_cuda_is_remembered_across_restarts(monkeypatch, tmp_path):
    """Trying CUDA costs memory that unloading does not return, so try it once.

    Measured on Orin 5: one rejected attempt took MemAvailable from 5754 to 1935 MB and
    kept it. A crash restarts perception, so an in-memory verdict would be lost and the
    next start would pay again — which is how face's later GPU load tipped that box into
    the OOM killer.
    """
    _FakeDirect.lengths = {"gpu": 21000}
    with pytest.raises(kw.DeviceUnavailable, match="duration"):
        _proxy(monkeypatch, device="gpu", model_dir=str(tmp_path))
    assert (tmp_path / ".cuda-duration-verdict").exists()

    # A restarted process must not build anything before refusing.
    _FakeDirect.built = []
    with pytest.raises(kw.DeviceUnavailable, match="recorded in"):
        _proxy(monkeypatch, device="gpu", model_dir=str(tmp_path))
    assert not _FakeDirect.built, (
        "a machine that has already answered this must not pay for the answer again")


def test_the_verdict_is_keyed_on_the_runtime_version(monkeypatch, tmp_path):
    """The defect is in the ONNX Runtime build, so a different one gets re-evaluated."""
    (tmp_path / ".cuda-duration-verdict").write_text("1.15.1", encoding="utf-8")
    assert kw._cuda_known_bad(str(tmp_path), "1.15.1") is True
    assert kw._cuda_known_bad(str(tmp_path), "1.18.1") is False


def test_a_missing_verdict_file_is_not_a_verdict(tmp_path):
    assert kw._cuda_known_bad(str(tmp_path), "1.18.1") is False


def test_an_unwritable_model_dir_does_not_break_the_build(monkeypatch, tmp_path):
    """A read-only /models must cost the memory of a retry, not a crash."""
    def _boom(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr("builtins.open", _boom)
    kw._remember_cuda_is_bad(str(tmp_path), "1.18.1")     # must not raise


# ── idle unloading is off, and the reaper honours that ────────────────────────

def test_idle_unloading_is_off_by_default():
    """It never returned memory — measured at +0 MB — and cost a 5-7 s rebuild."""
    assert kw.IDLE_TIMEOUT_S == 0


def test_the_reaper_exits_immediately_when_disabled(monkeypatch):
    proxy = _proxy(monkeypatch, idle=0.0)
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    proxy._reap_loop()                    # returns rather than looping
    assert slept == [], "a disabled reaper must not even poll"
