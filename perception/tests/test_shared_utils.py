"""
Host-side unit tests for the shared perception utils (no Jetson required).

Run from the repo root:
    python -m pytest perception/tests -q

`tensorrt` and `libcudart` are replaced by small fakes so the engine
bookkeeping (profile selection, buffer growth, output ordering, close) and
the verified-bundle downloader can be exercised anywhere.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from utils import model_downloader  # noqa: E402
from utils.latest_frame import LatestFrame  # noqa: E402
from utils import tensorrt_runtime as trt_rt  # noqa: E402


# ── tensorrt_family ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "version, family",
    [("8.5.2.2", "jp511"), ("8.6.1", "jp511"), ("10.3.0", "jp61"), ("10.7.0.23", "jp61")],
)
def test_tensorrt_family_from_version(version, family):
    assert trt_rt.tensorrt_family(version) == family


@pytest.mark.parametrize("version", ["7.2.3", "9.0.0", "", "abc"])
def test_tensorrt_family_rejects_unknown(version):
    with pytest.raises(trt_rt.TensorRTError):
        trt_rt.tensorrt_family(version)


@pytest.mark.parametrize("version", ["11.0.0", "12.2.1"])
def test_tensorrt_family_warns_for_a_major_newer_than_any_bundle(version, caplog):
    """A newer major still resolves, but must not do so silently.

    Refusing would take a device down on a JetPack upgrade, and the engines may
    well still load. But plans are not guaranteed portable across TensorRT
    majors, so without the warning a mismatch surfaces later as an unexplained
    deserialization failure. Needs PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 — caplog
    captures nothing inside a perception image otherwise (see perception/README).
    """
    with caplog.at_level("WARNING"):
        assert trt_rt.tensorrt_family(version) == "jp61"
    assert any("not portable across majors" in r.getMessage()
               for r in caplog.records), caplog.records


@pytest.mark.parametrize("version", ["8.5.2.2", "10.3.0"])
def test_tensorrt_family_is_quiet_for_a_known_major(version, caplog):
    with caplog.at_level("WARNING"):
        trt_rt.tensorrt_family(version)
    assert not [r for r in caplog.records if "tensorrt" in r.getMessage().lower()]


def test_normalize_family_aliases():
    assert trt_rt.normalize_family("61") == "jp61"
    assert trt_rt.normalize_family("JP511") == "jp511"
    assert trt_rt.normalize_family("511") == "jp511"
    assert trt_rt.normalize_family("bogus") is None
    assert trt_rt.normalize_family(None) is None


# ── read_engine_file ─────────────────────────────────────────────────────────

def test_read_engine_file_strips_ultralytics_header(tmp_path):
    payload = b"\x00\x01\x02engine-bytes"
    meta = b'{"task": "segment", "names": {"0": "person"}}'
    path = tmp_path / "x.engine"
    path.write_bytes(len(meta).to_bytes(4, "little") + meta + payload)
    metadata, serialized = trt_rt.read_engine_file(path)
    assert metadata["task"] == "segment"
    assert serialized == payload


def test_read_engine_file_plain_engine(tmp_path):
    payload = bytes(range(64))
    path = tmp_path / "plain.engine"
    path.write_bytes(payload)
    metadata, serialized = trt_rt.read_engine_file(path)
    assert metadata == {}
    assert serialized == payload


# ── fake TensorRT + CUDA ─────────────────────────────────────────────────────

class _FakeDataType:
    FLOAT, HALF, INT8, INT32, BOOL, UINT8 = range(6)


class _FakeIOMode:
    INPUT, OUTPUT = 0, 1


class _FakeEngine:
    """Single input "images" [-1,3,-1,-1] with two profiles, two outputs."""

    def __init__(self, static=False):
        self.static = static
        self.num_io_tensors = 3
        self.names = ["images", "out_a", "out_b"]
        self.num_optimization_profiles = 2
        self.profiles = [
            ((1, 3, 32, 32), (1, 3, 64, 64), (4, 3, 128, 128)),
            ((1, 3, 128, 128), (2, 3, 256, 256), (8, 3, 512, 512)),
        ]

    def get_tensor_name(self, i):
        return self.names[i]

    def get_tensor_mode(self, name):
        return _FakeIOMode.INPUT if name == "images" else _FakeIOMode.OUTPUT

    def get_tensor_dtype(self, name):
        return {"images": _FakeDataType.FLOAT, "out_a": _FakeDataType.HALF,
                "out_b": _FakeDataType.INT32}[name]

    def get_tensor_shape(self, name):
        if name == "images":
            return (1, 3, 64, 64) if self.static else (-1, 3, -1, -1)
        return (-1, 8) if name == "out_a" else (-1, 1)

    def get_tensor_profile_shape(self, name, index):
        return self.profiles[index]

    def create_execution_context(self):
        return _FakeContext(self)


class _FakeContext:
    def __init__(self, engine):
        self.engine = engine
        self.shape = None
        self.profile = 0
        self.addresses = {}
        self.executions = 0
        self.profile_switches = 0

    def set_optimization_profile_async(self, index, stream):
        self.profile = index
        self.profile_switches += 1
        return True

    def set_input_shape(self, name, shape):
        self.shape = tuple(shape)
        return True

    def get_tensor_shape(self, name):
        if name == "images":
            return self.shape
        n = self.shape[0]
        return (n, 8) if name == "out_a" else (n, 1)

    def set_tensor_address(self, name, pointer):
        self.addresses[name] = pointer

    def execute_async_v3(self, stream):
        self.executions += 1
        return True


class _FakeRuntime:
    def __init__(self, logger):
        pass

    def deserialize_cuda_engine(self, data):
        return _FakeEngine(static=(data == b"static"))


class _FakeLogger:
    WARNING = 1

    def __init__(self, level):
        pass


def _install_fake_tensorrt(monkeypatch, version="8.5.2.2"):
    module = types.ModuleType("tensorrt")
    module.__version__ = version
    module.DataType = _FakeDataType
    module.TensorIOMode = _FakeIOMode
    module.Runtime = _FakeRuntime
    module.Logger = _FakeLogger
    module.nptype = lambda dtype: np.float32
    monkeypatch.setitem(sys.modules, "tensorrt", module)
    return module


class _FakeCudart:
    """Records cudaMalloc/cudaFree so buffer growth and cleanup can be asserted."""

    def __init__(self):
        self.allocs = {}
        self.freed = []
        self.streams_created = 0
        self.streams_destroyed = 0
        self.copies = []
        self._next = 0x1000

    # ctypes-style attribute assignments must be accepted
    class _Fn:
        def __init__(self, impl):
            self.impl = impl
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.impl(*args)

    def __getattr__(self, name):
        impl = getattr(self, "_" + name, None)
        if impl is None:
            raise AttributeError(name)
        fn = _FakeCudart._Fn(impl)
        setattr(self, name, fn)
        return fn

    def _cudaGetErrorString(self, code):
        return b"fake error"

    def _cudaSetDevice(self, device):
        return 0

    def _cudaStreamCreateWithFlags(self, ptr, flags):
        self.streams_created += 1
        ptr._obj.value = 0xBEEF
        return 0

    def _cudaStreamDestroy(self, stream):
        self.streams_destroyed += 1
        return 0

    def _cudaStreamSynchronize(self, stream):
        return 0

    def _cudaMalloc(self, ptr, size):
        pointer = self._next
        self._next += 0x1000
        self.allocs[pointer] = int(size.value if hasattr(size, "value") else size)
        ptr._obj.value = pointer
        return 0

    def _cudaFree(self, pointer):
        value = pointer.value if hasattr(pointer, "value") else pointer
        self.freed.append(value)
        return 0

    def _cudaMemcpyAsync(self, dst, src, size, kind, stream):
        self.copies.append(kind)
        return 0


@pytest.fixture
def fake_cuda(monkeypatch):
    fake = _FakeCudart()
    monkeypatch.setattr(trt_rt.ctypes, "CDLL", lambda name: fake)
    monkeypatch.setattr(trt_rt.ctypes.util, "find_library", lambda name: "libcudart.so")
    return fake


def test_engine_dynamic_profiles_and_buffer_growth(tmp_path, monkeypatch, fake_cuda):
    _install_fake_tensorrt(monkeypatch)
    path = tmp_path / "dyn.engine"
    path.write_bytes(b"dynamic")
    engine = trt_rt.TensorRTEngine(path)
    assert engine.input_name == "images"
    assert engine.output_names == ["out_a", "out_b"]
    assert not engine.is_static
    assert engine.input_shape is None
    assert engine.optimization_shape == (1, 3, 64, 64)
    assert engine.input_dtype == np.float32
    assert engine.output_dtypes["out_a"] == np.float16

    # profile 0 covers 64x64; profile 1 covers 256x256; nothing covers 1024.
    assert engine.select_profile((1, 3, 64, 64)) == 0
    assert engine.select_profile((2, 3, 256, 256)) == 1
    with pytest.raises(trt_rt.TensorRTShapeError):
        engine.select_profile((1, 3, 1024, 1024))

    outputs = engine.infer(np.zeros((1, 3, 64, 64), dtype=np.uint8))
    assert [o.shape for o in outputs] == [(1, 8), (1, 1)]
    assert outputs[0].dtype == np.float16 and outputs[1].dtype == np.int32
    allocs_after_first = dict(fake_cuda.allocs)
    assert len(allocs_after_first) == 3  # input + 2 outputs

    # Same shape again: no new allocations, no profile switch.
    engine.infer(np.zeros((1, 3, 64, 64), dtype=np.float32))
    assert fake_cuda.allocs == allocs_after_first
    assert engine._context.profile_switches == 0

    # Larger batch on the other profile: input buffer grows, old one freed.
    outputs = engine.infer(np.zeros((3, 3, 256, 256), dtype=np.float32))
    assert [o.shape for o in outputs] == [(3, 8), (3, 1)]
    assert engine._context.profile == 1
    assert len(fake_cuda.freed) >= 1

    engine.close()
    assert set(fake_cuda.freed) >= set(allocs_after_first)
    assert fake_cuda.streams_destroyed == 1
    with pytest.raises(trt_rt.TensorRTError):
        engine.infer(np.zeros((1, 3, 64, 64), dtype=np.float32))
    engine.close()  # idempotent


def test_engine_static_shape(tmp_path, monkeypatch, fake_cuda):
    _install_fake_tensorrt(monkeypatch, version="10.3.0")
    path = tmp_path / "static.engine"
    path.write_bytes(b"static")
    engine = trt_rt.TensorRTEngine(path)
    assert engine.is_static
    assert engine.input_shape == (1, 3, 64, 64)
    with pytest.raises(trt_rt.TensorRTShapeError):
        engine.infer(np.zeros((1, 3, 32, 32), dtype=np.float32))
    outputs = engine.infer(np.zeros((1, 3, 64, 64), dtype=np.float32))
    assert len(outputs) == 2
    engine.close()


def test_engine_rejects_multi_input(tmp_path, monkeypatch, fake_cuda):
    module = _install_fake_tensorrt(monkeypatch)

    class TwoInputEngine(_FakeEngine):
        def get_tensor_mode(self, name):
            return _FakeIOMode.INPUT if name in ("images", "out_a") else _FakeIOMode.OUTPUT

    monkeypatch.setattr(
        module.Runtime, "deserialize_cuda_engine",
        lambda self, data: TwoInputEngine(),
    )
    path = tmp_path / "two.engine"
    path.write_bytes(b"x")
    with pytest.raises(trt_rt.TensorRTError, match="exactly one input"):
        trt_rt.TensorRTEngine(path)
    # resources released even though construction failed
    assert fake_cuda.streams_destroyed == fake_cuda.streams_created == 1


def test_trt_dtype_to_numpy_without_nptype():
    class DT:
        FLOAT, HALF, INT8, INT32, BOOL, UINT8, INT64 = range(7)

    module = types.SimpleNamespace(DataType=DT, nptype=lambda d: (_ for _ in ()).throw(AttributeError))
    assert trt_rt.trt_dtype_to_numpy(module, DT.BOOL) == np.bool_
    assert trt_rt.trt_dtype_to_numpy(module, DT.INT64) == np.int64
    assert trt_rt.trt_dtype_to_numpy(module, DT.UINT8) == np.uint8


# ── LatestFrame ──────────────────────────────────────────────────────────────

def test_latest_frame_keeps_only_newest():
    buffer = LatestFrame()
    assert buffer.pop(timeout=0) is None
    for index in range(10):
        buffer.push(index)
    assert buffer.pop(timeout=0) == 9
    assert buffer.dropped == 9
    assert buffer.pop(timeout=0) is None


def test_latest_frame_wakes_waiter_and_close():
    buffer = LatestFrame()
    seen = []

    def worker():
        seen.append(buffer.pop(timeout=5))
        seen.append(buffer.pop(timeout=5))

    thread = threading.Thread(target=worker)
    thread.start()
    time.sleep(0.05)
    buffer.push("frame")
    time.sleep(0.05)
    buffer.close()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert seen == ["frame", None]
    buffer.push("ignored")
    assert buffer.pop(timeout=0) is None and buffer.closed


# ── model_downloader verified bundles ────────────────────────────────────────

def _manifest(payloads: dict[str, bytes]) -> dict:
    return {
        name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in payloads.items()
    }


class _FakeResponse:
    def __init__(self, data):
        self._data = data
        self._offset = 0

    def read(self, size):
        chunk = self._data[self._offset : self._offset + size]
        self._offset += size
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, payloads: dict[str, bytes], calls: list):
    def fake_urlopen(url, timeout=0):
        calls.append(url)
        name = url.rsplit("/", 1)[1]
        return _FakeResponse(payloads[name])

    monkeypatch.setattr(model_downloader, "urlopen", fake_urlopen)
    monkeypatch.setattr(model_downloader.time, "sleep", lambda s: None)


def test_bundle_matches_requires_size_and_sha(tmp_path):
    payloads = {"det.engine": b"detector", "keys.txt": b"a\nb\n"}
    manifest = _manifest(payloads)
    for name, data in payloads.items():
        (tmp_path / name).write_bytes(data)
    assert model_downloader._bundle_matches(str(tmp_path), manifest)

    # non-empty but truncated file must NOT count as a valid cache
    (tmp_path / "det.engine").write_bytes(b"det")
    assert not model_downloader._bundle_matches(str(tmp_path), manifest)
    # same size, different content
    (tmp_path / "det.engine").write_bytes(b"detectoX")
    assert not model_downloader._bundle_matches(str(tmp_path), manifest)


def test_ensure_verified_bundle_downloads_repairs_and_reuses(tmp_path, monkeypatch):
    payloads = {"det.engine": b"detector-bytes", "keys.txt": b"a\nb\n"}
    manifest = _manifest(payloads)
    calls: list[str] = []
    _serve(monkeypatch, payloads, calls)
    model_dir = tmp_path / "ocr"

    paths = model_downloader.ensure_verified_bundle(
        "ocr/jp511", str(model_dir), "https://example/base/", manifest
    )
    assert set(paths) == set(payloads)
    assert (model_dir / "det.engine").read_bytes() == payloads["det.engine"]
    assert calls == ["https://example/base/det.engine", "https://example/base/keys.txt"]
    assert not [p for p in model_dir.iterdir() if p.name.startswith(".ocr") and p.is_dir()]

    # valid cache → no network
    calls.clear()
    model_downloader.ensure_verified_bundle(
        "ocr/jp511", str(model_dir), "https://example/base/", manifest
    )
    assert calls == []

    # corrupted engine (nonempty, wrong hash) → re-downloaded
    (model_dir / "det.engine").write_bytes(b"detector-bytes"[:-1] + b"X")
    model_downloader.ensure_verified_bundle(
        "ocr/jp511", str(model_dir), "https://example/base/", manifest
    )
    assert calls  # network used
    assert (model_dir / "det.engine").read_bytes() == payloads["det.engine"]


def test_ensure_verified_bundle_retries_then_fails(tmp_path, monkeypatch):
    payloads = {"det.engine": b"detector-bytes"}
    manifest = _manifest(payloads)
    calls: list[str] = []
    _serve(monkeypatch, {"det.engine": b"garbage"}, calls)  # never matches sha
    with pytest.raises(RuntimeError, match="failed to download det.engine"):
        model_downloader.ensure_verified_bundle(
            "ocr/jp511", str(tmp_path / "m"), "https://example/base", manifest
        )
    assert len(calls) == 3
    assert not (tmp_path / "m" / "det.engine").exists()


def test_select_bundle_family(tmp_path, monkeypatch):
    bundles = {
        "jp511": {"base_url": "https://x/jp511", "files": {}},
        "jp61": {"base_url": "https://x/jp61", "files": {}},
    }

    # explicit override wins and accepts aliases
    assert model_downloader.select_bundle_family(bundles, family="61") == "jp61"
    assert model_downloader.select_bundle_family(bundles, family="JP511") == "jp511"

    # runtime TensorRT decides otherwise
    _install_fake_tensorrt(monkeypatch, version="8.5.2.2")
    assert model_downloader.select_bundle_family(bundles) == "jp511"
    _install_fake_tensorrt(monkeypatch, version="10.3.0")
    assert model_downloader.select_bundle_family(bundles) == "jp61"

    # unknown alias / missing family raise
    with pytest.raises(ValueError):
        model_downloader.select_bundle_family(bundles, family="jp7")
    with pytest.raises(RuntimeError):
        model_downloader.select_bundle_family({"jp61": bundles["jp61"]}, family="511")


# ── SampledLogGate ───────────────────────────────────────────────────────────

def test_sampled_log_gate_transitions_and_sampling():
    from utils.log_sampling import SampledLogGate

    gate = SampledLogGate(every=100)
    # first occurrence logs, and is a transition
    assert gate.check("ok") == (True, True, 1)
    # steady state: 2..99 stay quiet, 100th logs
    decisions = [gate.check("ok") for _ in range(2, 101)]
    assert all(not d[0] for d in decisions[:-1])
    assert decisions[-1] == (True, False, 100)
    # outcome change logs immediately regardless of counters
    assert gate.check("error:Timeout") == (True, True, 1)
    # flapping back is a transition again
    assert gate.check("ok") == (True, True, 1)


def test_escape_log_text_caps_and_escapes():
    from utils.log_sampling import escape_log_text

    assert escape_log_text("plain error") == "plain error"
    # control characters and ANSI escapes become visible escapes
    assert escape_log_text("bad\x00\x1b[31mvalue\n") == "bad\\x00\\x1b[31mvalue\\n"
    # long values are capped
    out = escape_log_text("x" * 500, cap=200)
    assert len(out) == 203 and out.endswith("...")


def test_engine_close_waits_for_inflight_infer(monkeypatch, tmp_path):
    """close() must synchronize with infer(): freeing CUDA buffers under an
    in-flight execution is a use-after-free (bot P1). close() blocks until the
    running infer finishes; a later infer raises cleanly."""
    import threading, time
    _install_fake_tensorrt(monkeypatch)
    fake_cudart = _FakeCudart()
    monkeypatch.setattr(trt_rt.ctypes, "CDLL", lambda name: fake_cudart)
    monkeypatch.setattr(trt_rt.ctypes.util, "find_library", lambda name: "libcudart.so")

    engine_path = tmp_path / "e.engine"
    engine_path.write_bytes(b"static")
    engine = trt_rt.TensorRTEngine(str(engine_path))

    entered = threading.Event()
    release = threading.Event()
    def slow_execute(*a, **k):
        entered.set()
        release.wait(timeout=5)
        return True

    monkeypatch.setattr(engine._context, "execute_async_v3", slow_execute)

    freed_during_infer = []
    worker_error = []

    def run_infer():
        try:
            engine.infer(np.zeros((1, 3, 64, 64), dtype=np.float32))
        except Exception as error:  # noqa: BLE001
            worker_error.append(error)

    t1 = threading.Thread(target=run_infer)
    t1.start()
    assert entered.wait(timeout=5), "infer never reached execution"

    closer = threading.Thread(target=engine.close)
    closer.start()
    time.sleep(0.2)
    # close must NOT have freed anything while infer is still executing
    freed_during_infer = list(fake_cudart.freed)
    assert not freed_during_infer, "CUDA buffers freed under in-flight infer"
    assert closer.is_alive(), "close() returned while infer was executing"

    release.set()
    t1.join(timeout=5)
    closer.join(timeout=5)
    assert not closer.is_alive()
    assert fake_cudart.freed, "buffers were never freed after close"

    with pytest.raises(trt_rt.TensorRTError):
        engine.infer(np.zeros((1, 3, 64, 64), dtype=np.float32))


def test_require_models_subpath():
    from utils.model_downloader import require_models_subpath

    assert require_models_subpath("/models/ocr/x") == "/models/ocr/x"
    assert require_models_subpath("/models") == "/models"
    # traversal and out-of-tree paths are rejected
    for bad in ("/etc/cron.d", "/models/../etc", "relative/dir", "/modelsevil"):
        with pytest.raises(ValueError):
            require_models_subpath(bad)


def test_require_models_subpath_rejects_symlink_escape(tmp_path):
    """A lexical check passes /models/link while link points outside the tree;
    every later makedirs/open/replace would follow it as root."""
    from utils.model_downloader import require_models_subpath

    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    # symlinked directory inside the tree
    (root / "link").symlink_to(outside)
    with pytest.raises(ValueError):
        require_models_subpath(str(root / "link"), root=str(root))
    # and a not-yet-created child under that symlink
    with pytest.raises(ValueError):
        require_models_subpath(str(root / "link" / "bundle"), root=str(root))

    # a symlink that stays inside the tree is still allowed
    (root / "inner").mkdir()
    (root / "alias").symlink_to(root / "inner")
    assert require_models_subpath(str(root / "alias"), root=str(root)) == str(
        (root / "inner").resolve()
    )

    # plain paths, existing or not, keep working
    assert require_models_subpath(str(root / "new" / "bundle"), root=str(root)) == str(
        root.resolve() / "new" / "bundle"
    )


def test_ensure_model_publishes_check_file_only_when_complete(tmp_path, monkeypatch):
    """A concurrent reader must never see a half-written model.

    ensure_model used to urlretrieve() straight onto check_file, so a second
    caller saw the file the moment the transfer began and loaded a truncated
    model — on device: "Load model from .../vocos-16khz-univ.onnx failed:
    Protobuf parsing failed" while the log showed that file still at 30%.
    """
    from utils import model_downloader

    model_dir = tmp_path / "sherpa"
    check_file = "vocos-16khz-univ.onnx"
    seen_midway = []

    def fake_urlretrieve(url, dest, reporthook=None):
        # Mid-transfer: whatever a parallel caller can see must not be the
        # check_file, and the partial bytes must live somewhere else.
        with open(dest, "wb") as handle:
            handle.write(b"partial")
            handle.flush()
        seen_midway.append(sorted(p.name for p in model_dir.iterdir()))
        with open(dest, "wb") as handle:
            handle.write(b"complete-onnx-payload")

    monkeypatch.setattr(model_downloader, "urlretrieve", fake_urlretrieve)
    monkeypatch.setitem(
        model_downloader.MODELS, "tts_vocoder",
        {"url": "https://example.invalid/vocos.onnx", "check_file": check_file,
         "single_file": True},
    )

    model_downloader.ensure_model("tts_vocoder", str(model_dir))

    assert check_file not in seen_midway[0], seen_midway[0]
    assert (model_dir / check_file).read_bytes() == b"complete-onnx-payload"
    # No .part leftovers.
    assert not [p.name for p in model_dir.iterdir() if p.name.endswith(".part")]


def test_ensure_model_archive_is_published_atomically(tmp_path, monkeypatch):
    """Same guarantee for the archive path: extract to staging, then move in."""
    import tarfile
    from utils import model_downloader

    model_dir = tmp_path / "asr"
    archive = tmp_path / "src.tar.bz2"
    payload = tmp_path / "build"
    (payload / "sub").mkdir(parents=True)
    (payload / "tokens.txt").write_text("tokens")
    (payload / "sub" / "encoder.onnx").write_bytes(b"weights")
    with tarfile.open(archive, "w:bz2") as handle:
        handle.add(payload / "tokens.txt", arcname="model/tokens.txt")
        handle.add(payload / "sub" / "encoder.onnx", arcname="model/sub/encoder.onnx")

    seen_midway = []

    def fake_urlretrieve(url, dest, reporthook=None):
        seen_midway.append(sorted(p.name for p in model_dir.iterdir()))
        with open(archive, "rb") as src, open(dest, "wb") as out:
            out.write(src.read())

    monkeypatch.setattr(model_downloader, "urlretrieve", fake_urlretrieve)
    monkeypatch.setitem(
        model_downloader.MODELS, "asr",
        {"url": "https://example.invalid/asr.tar.bz2", "check_file": "tokens.txt"},
    )

    model_downloader.ensure_model("asr", str(model_dir))

    assert "tokens.txt" not in seen_midway[0]
    assert (model_dir / "tokens.txt").read_text() == "tokens"
    assert (model_dir / "sub" / "encoder.onnx").read_bytes() == b"weights"

    # Second call is a no-op that does not re-download.
    seen_midway.clear()
    model_downloader.ensure_model("asr", str(model_dir))
    assert seen_midway == []
