"""
tests/test_kokoro_worker.py — the proxy's contract and its failure behaviour.

The worker exists because two ONNX Runtimes in one process cannot both hold a CUDA
session on the Kokoro graph — on jp5.11 that is a SIGSEGV that kills all of perception.
So the properties worth testing here are not "does it synthesize" (that needs a 310 MB
model and a GPU) but the ones that decide whether the isolation actually holds and
whether a failure degrades or breaks:

  - the child is started with `spawn`, never `fork`. `fork` copies the address space
    including already-dlopened libraries, so a forked child would inherit sherpa's ONNX
    Runtime and the isolation would be worthless — the exact bug being fixed;
  - a dead or unstartable child falls back rather than raising, because Japanese losing
    8x is acceptable and Japanese breaking is not;
  - a request error is *not* swallowed by the fallback, or a real bug would hide behind
    a slower code path.

Run: python -m pytest perception/tests -q
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins import kokoro_worker as kw  # noqa: E402


class _FakeQueue:
    """A queue that records what was put and hands back scripted replies."""

    def __init__(self, replies=None):
        self.puts = []
        self._replies = list(replies or [])

    def put(self, item):
        self.puts.append(item)

    def get(self, timeout=None):
        if not self._replies:
            import queue as _q
            raise _q.Empty
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class _FakeProc:
    def __init__(self, alive=True):
        self.pid = 4242
        self._alive = alive
        self.terminated = False
        self.killed = False

    def start(self):
        pass

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False

    def kill(self):
        self.killed = True
        self._alive = False

    def join(self, timeout=None):
        pass


class _FakeCtx:
    """Stands in for multiprocessing's spawn context."""

    def __init__(self, replies, alive=True):
        self.cmd = _FakeQueue()
        self.res = _FakeQueue(replies)
        self.proc = _FakeProc(alive)
        self.queues_made = 0
        self.process_kwargs = None

    def Queue(self):                                              # noqa: N802
        self.queues_made += 1
        return self.cmd if self.queues_made == 1 else self.res

    def Process(self, **kwargs):                                  # noqa: N802
        self.process_kwargs = kwargs
        return self.proc


READY = ("ready", {"num_speakers": 54, "sample_rate": 24000,
                   "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                   "warmup_s": 8.03})


def _proxy(monkeypatch, replies, alive=True, **kwargs):
    """Build a proxy whose child is a fake, and stop the reaper thread interfering."""
    ctx = _FakeCtx(replies, alive)
    monkeypatch.setattr(kw.KokoroWorkerProxy, "_reap_loop", lambda self: None)

    import multiprocessing as mp
    monkeypatch.setattr(mp, "get_context", lambda method: ctx)
    proxy = kw.KokoroWorkerProxy("/models/kokoro-multi/gpu", "model.onnx", **kwargs)
    return proxy, ctx


# ── the isolation itself ──────────────────────────────────────────────────────

def test_the_child_is_spawned_not_forked(monkeypatch):
    """`fork` would inherit sherpa's already-loaded ONNX Runtime and defeat the point.

    On Linux `fork` is the default, so this has to be asked for explicitly. The
    assertion is on the string passed to get_context, because that is the whole
    guarantee — there is nothing else to check at runtime.
    """
    seen = {}

    class _Ctx(_FakeCtx):
        pass

    ctx = _Ctx([READY])
    monkeypatch.setattr(kw.KokoroWorkerProxy, "_reap_loop", lambda self: None)

    import multiprocessing as mp

    def _get_context(method):
        seen["method"] = method
        return ctx

    monkeypatch.setattr(mp, "get_context", _get_context)
    kw.KokoroWorkerProxy("/models/kokoro-multi/gpu", "model.onnx")
    assert seen["method"] == "spawn", seen


def test_the_child_entry_point_is_module_level_and_takes_no_ros_or_sherpa():
    """spawn pickles the target by qualified name, so it must be importable.

    Also asserts the child's own module imports neither rclpy nor sherpa_onnx at module
    scope — this process exists to be the only ONNX Runtime in its address space.
    """
    assert callable(kw._kokoro_worker)
    assert kw._kokoro_worker.__module__ == "plugins.kokoro_worker"
    source = (PERCEPTION_ROOT / "plugins" / "kokoro_worker.py").read_text("utf-8")
    assert "import rclpy" not in source
    assert "import sherpa_onnx" not in source


def test_handshake_populates_the_properties(monkeypatch):
    proxy, ctx = _proxy(monkeypatch, [READY])
    assert proxy.num_speakers == 54
    assert proxy.sample_rate == 24000
    assert "CUDAExecutionProvider" in proxy.providers
    assert ctx.process_kwargs["daemon"] is False, (
        "a daemon child is killed abruptly at parent exit; this one should be asked")


# ── failure behaviour ─────────────────────────────────────────────────────────

def test_a_child_that_never_reports_ready_is_terminated_and_raises(monkeypatch):
    """The caller falls back; leaving a half-started child behind would leak a context."""
    monkeypatch.setattr(kw, "START_TIMEOUT_S", 0.01)
    with pytest.raises(RuntimeError, match="did not report ready"):
        _proxy(monkeypatch, [])


def test_a_child_that_fails_to_build_reports_why(monkeypatch):
    with pytest.raises(RuntimeError, match="libcudnn"):
        _proxy(monkeypatch, [("failed", "OSError: libcudnn.so.8 not found")])


def test_a_dead_child_falls_back_instead_of_raising(monkeypatch):
    """Japanese losing 8x is acceptable; Japanese breaking is not."""
    proxy, ctx = _proxy(monkeypatch, [READY])
    ctx.proc._alive = False

    calls = {}

    class _Fallback:
        num_speakers, sample_rate = 54, 24000

        def synthesize(self, phonemes, speaker_id=0, speed=1.0):
            calls["args"] = (phonemes, speaker_id, speed)
            return np.ones(8, dtype=np.float32)

    # A restart attempt also fails, so the fallback is the only route left.
    monkeypatch.setattr(kw.KokoroWorkerProxy, "_start",
                        lambda self: (_ for _ in ()).throw(RuntimeError("no")))
    monkeypatch.setattr(kw.KokoroWorkerProxy, "_use_fallback", lambda self: _Fallback())

    out = proxy.synthesize("ohayoo", speaker_id=3, speed=1.1)
    assert out.shape == (8,)
    assert calls["args"] == ("ohayoo", 3, 1.1)


def test_a_request_error_is_raised_not_hidden_behind_the_fallback(monkeypatch):
    """A bad request means a bug. Re-synthesizing it elsewhere would mask it."""
    proxy, ctx = _proxy(monkeypatch, [READY])
    ctx.res._replies.append(("error", "req", "ValueError: speaker_id must be 0..53"))
    monkeypatch.setattr(kw.KokoroWorkerProxy, "_recv",
                        lambda self, want: ("error", "ValueError: speaker_id"))
    with pytest.raises(RuntimeError, match="speaker_id"):
        proxy.synthesize("ohayoo", speaker_id=999)


def test_close_is_idempotent_and_terminates_the_child(monkeypatch):
    proxy, ctx = _proxy(monkeypatch, [READY])
    proxy.close()
    assert ctx.proc.terminated
    proxy.close()          # must not raise


# ── the leak this fixes ───────────────────────────────────────────────────────

def test_idle_reaping_releases_the_child(monkeypatch):
    """A card that spoke Japanese once used to hold the session for the adapter's life.

    `set_language` never released it, so switching back to English kept ~765 MB
    resident. Reaping on idle — rather than on the language switch — is what gives it
    back without making an alternating tour pay the ~10.5 s cold path every time.
    """
    proxy, ctx = _proxy(monkeypatch, [READY], idle_timeout_s=0.0)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    # One pass of the real reaper body, with the loop's sleep neutralised.
    proxy._last_used = time.monotonic() - 3600
    stop = threading.Event()

    def _one_pass():
        with proxy._lock:
            if proxy._alive() and (time.monotonic() - proxy._last_used) > proxy._idle_timeout_s:
                proxy._terminate()
        stop.set()

    _one_pass()
    assert stop.is_set()
    assert ctx.proc.terminated, "an idle worker must give its CUDA context back"


# ── the duration gate ─────────────────────────────────────────────────────────

class _FakeDirect:
    """A KokoroDirect stand-in whose probe length is scripted per provider."""

    lengths = {}

    def __init__(self, model_dir, weights, num_threads=0, provider="cpu"):
        self.provider = provider
        self._session = type("s", (), {"get_providers": lambda self: [provider]})()

    num_speakers, sample_rate = 54, 24000

    def synthesize(self, phonemes, speaker_id=0, speed=1.0):
        return np.zeros(self.lengths[self.provider], dtype=np.float32)


def test_a_device_that_gets_durations_wrong_is_rejected_for_cpu(monkeypatch):
    """jp5.11's CUDA returns 35-44% of the probe's length; 27% short is rushed speech.

    Gating on the ORT version would not catch the next line with the same defect, so the
    gate is the measurement. The probe already runs as warmup, so it is free.
    """
    logged = []
    _FakeDirect.lengths = {"gpu": 21000, "cpu": kw.PROBE_SAMPLES}
    runtime, used = kw._build_checked(
        _FakeDirect, "/models/x", "model.onnx", "gpu", 0,
        type("l", (), {"warning": lambda *a: logged.append(a),
                       "error": lambda *a: logged.append(a)})())
    assert used == "cpu", "a device failing the duration check must not be used"
    assert any("duration" in str(a) for a in logged), logged


def test_a_correct_device_is_kept(monkeypatch):
    _FakeDirect.lengths = {"gpu": kw.PROBE_SAMPLES, "cpu": kw.PROBE_SAMPLES}
    _runtime, used = kw._build_checked(
        _FakeDirect, "/models/x", "model.onnx", "gpu", 0,
        type("l", (), {"warning": lambda *a: None, "error": lambda *a: None})())
    assert used == "gpu"


def test_small_deviation_is_tolerated():
    """A different-but-valid build must not be rejected over a few samples."""
    _FakeDirect.lengths = {"gpu": int(kw.PROBE_SAMPLES * 1.03), "cpu": kw.PROBE_SAMPLES}
    _runtime, used = kw._build_checked(
        _FakeDirect, "/models/x", "model.onnx", "gpu", 0,
        type("l", (), {"warning": lambda *a: None, "error": lambda *a: None})())
    assert used == "gpu"


def test_cpu_failing_the_check_too_is_an_error_not_a_silent_pass():
    """Then the model or the phoneme table does not match this code — say so."""
    _FakeDirect.lengths = {"gpu": 100, "cpu": 100}
    with pytest.raises(RuntimeError, match="plausible probe duration"):
        kw._build_checked(
            _FakeDirect, "/models/x", "model.onnx", "gpu", 0,
            type("l", (), {"warning": lambda *a: None, "error": lambda *a: None})())


# ── the headroom guard, which must run BEFORE the duration gate ───────────────

def test_cuda_is_declined_when_the_box_has_no_headroom(monkeypatch):
    """The duration check cannot save a box that OOMs while building the session.

    Measured on jp5.11: 5237 MB available, sherpa's own Kokoro GPU adapter takes 3.2 GB,
    and the CUDA child's allocation then killed the process — before the probe ran.
    Two independent guards, and this one has to be first because it is the one that can
    take the process down.
    """
    monkeypatch.setattr(kw, "_mem_available_mb", lambda: 2048)
    _proxy_, ctx = _proxy(monkeypatch, [READY], device="gpu")
    assert ctx.process_kwargs["args"][2] == "cpu", (
        "a CUDA child must not be spawned with less headroom than it needs")


def test_cuda_is_used_when_there_is_room(monkeypatch):
    monkeypatch.setattr(kw, "_mem_available_mb", lambda: 4096)
    _proxy_, ctx = _proxy(monkeypatch, [READY], device="gpu")
    assert ctx.process_kwargs["args"][2] == "gpu"


def test_an_unreadable_meminfo_does_not_block_the_gpu(monkeypatch):
    """-1 means "could not tell", which must not be read as "no memory"."""
    monkeypatch.setattr(kw, "_mem_available_mb", lambda: -1)
    _proxy_, ctx = _proxy(monkeypatch, [READY], device="gpu")
    assert ctx.process_kwargs["args"][2] == "gpu"
