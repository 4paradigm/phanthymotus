"""
utils/tensorrt_runtime.py — Shared TensorRT / CUDA runtime for Jetson plugins.

Jetson + CUDA + TensorRT are platform capabilities of the perception base
image. Vision plugins (OCR, obstacle, ...) only ship their own engines and
pre/post-processing; everything below is the common part:

    JetPack family selection   → tensorrt_family()
    TensorRT ↔ numpy dtypes    → trt_dtype_to_numpy()
    engine bytes (+ metadata)  → read_engine_file()
    CUDA stream / buffers      → CudaRuntime
    deserialize / execute      → TensorRTEngine

Works with TensorRT 8.5 (JetPack 5.1.1) and TensorRT 10 (JetPack 6.x): both
expose the tensor-name based API (num_io_tensors, set_tensor_address,
execute_async_v3) that this module relies on. `tensorrt` itself is imported
lazily so importing this module on a non-Jetson host is harmless.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import logging
import math
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "TensorRTError",
    "TensorRTShapeError",
    "tensorrt_family",
    "tensorrt_version",
    "normalize_family",
    "trt_dtype_to_numpy",
    "read_engine_file",
    "CudaRuntime",
    "TensorRTEngine",
]


class TensorRTError(RuntimeError):
    """Raised for TensorRT / CUDA runtime failures."""


class TensorRTShapeError(TensorRTError, ValueError):
    """Raised when an input shape is not covered by any engine profile."""


# ── JetPack family selection ─────────────────────────────────────────────────

# TensorRT major → engine bundle family. Engines are not portable across
# TensorRT majors, so the family is decided by the TensorRT that is actually
# importable at runtime, never by a Docker build argument.
_TRT_FAMILIES = {8: "jp511", 10: "jp61"}
_FAMILY_ALIASES = {
    "511": "jp511", "jp511": "jp511", "jp5": "jp511",
    "61": "jp61", "jp61": "jp61", "jp6": "jp61",
}


def tensorrt_version() -> str:
    """Return the importable TensorRT version string (raises TensorRTError)."""
    try:
        import tensorrt
    except ImportError as error:
        raise TensorRTError("TensorRT is not available in this runtime") from error
    return str(getattr(tensorrt, "__version__", ""))


def tensorrt_family(version: str | None = None) -> str:
    """Map the runtime TensorRT version to an engine family ("jp511"/"jp61").

    Passing an explicit `version` skips importing tensorrt (used by tests and
    by manifests that want to describe both families).
    """
    text = tensorrt_version() if version is None else str(version)
    try:
        major = int(text.split(".", 1)[0])
    except ValueError as error:
        raise TensorRTError(f"Unsupported TensorRT version: {text or 'unknown'}") from error
    if major >= 10:
        return "jp61"
    family = _TRT_FAMILIES.get(major)
    if family is None:
        raise TensorRTError(f"Unsupported TensorRT version: {text}")
    return family


def normalize_family(value: object) -> str | None:
    """Normalize a user/env supplied family hint ("61", "jp511", ...) or None."""
    if value is None:
        return None
    return _FAMILY_ALIASES.get(str(value).strip().lower())


# ── dtype mapping ────────────────────────────────────────────────────────────

def trt_dtype_to_numpy(trt_module, dtype) -> np.dtype:
    """Map TensorRT dtypes to numpy without going through trt.nptype().

    TensorRT 8.x's nptype() still references np.bool, which NumPy 1.24 removed,
    so on the jp511 stack (TRT 8.5 + numpy 1.24.4) it raises AttributeError the
    moment an engine is loaded. Resolve the common dtypes ourselves and only
    fall back to nptype() for anything exotic.
    """
    mapping = {
        trt_module.DataType.FLOAT: np.float32,
        trt_module.DataType.HALF: np.float16,
        trt_module.DataType.INT8: np.int8,
        trt_module.DataType.INT32: np.int32,
        trt_module.DataType.BOOL: np.bool_,
    }
    for name in ("UINT8", "INT64"):
        member = getattr(trt_module.DataType, name, None)
        if member is not None:
            mapping[member] = getattr(np, name.lower())
    try:
        return np.dtype(mapping[dtype])
    except KeyError:
        return np.dtype(trt_module.nptype(dtype))


# ── engine file ──────────────────────────────────────────────────────────────

def read_engine_file(path: str | Path) -> tuple[dict, bytes]:
    """Read a serialized engine, stripping an optional JSON metadata header.

    Ultralytics-style exports prefix the engine with `<int32 size><json>`.
    Plain TensorRT engines are returned unchanged with `{}` metadata.
    """
    try:
        serialized = Path(path).read_bytes()
    except OSError as error:
        raise TensorRTError(f"TensorRT engine could not be read: {path}") from error
    if len(serialized) < 8:
        return {}, serialized
    metadata_size = int.from_bytes(serialized[:4], "little", signed=True)
    if metadata_size <= 0 or metadata_size > min(len(serialized) - 4, 1 << 20):
        return {}, serialized
    try:
        metadata = json.loads(serialized[4 : 4 + metadata_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}, serialized
    if not isinstance(metadata, dict) or "task" not in metadata:
        return {}, serialized
    return metadata, serialized[4 + metadata_size :]


# ── CUDA runtime ─────────────────────────────────────────────────────────────

class CudaRuntime:
    """Minimal checked wrapper around the CUDA Runtime C API (one stream)."""

    _MEMCPY_HOST_TO_DEVICE = 1
    _MEMCPY_DEVICE_TO_HOST = 2
    _STREAM_NON_BLOCKING = 1

    def __init__(self, device_id: int = 0):
        self._device_id = int(device_id)
        library = ctypes.util.find_library("cudart") or "libcudart.so"
        try:
            self._lib = ctypes.CDLL(library)
        except OSError as error:
            raise TensorRTError("CUDA Runtime library could not be loaded") from error
        self._bind()
        self._stream = ctypes.c_void_p()
        self._set_device()
        self._check(
            self._lib.cudaStreamCreateWithFlags(
                ctypes.byref(self._stream), self._STREAM_NON_BLOCKING
            ),
            "cudaStreamCreateWithFlags",
        )

    def _bind(self) -> None:
        lib = self._lib
        lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        lib.cudaSetDevice.argtypes = [ctypes.c_int]
        lib.cudaSetDevice.restype = ctypes.c_int
        lib.cudaStreamCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        lib.cudaStreamCreateWithFlags.restype = ctypes.c_int
        lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        lib.cudaStreamDestroy.restype = ctypes.c_int
        lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.cudaStreamSynchronize.restype = ctypes.c_int
        lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cudaMalloc.restype = ctypes.c_int
        lib.cudaFree.argtypes = [ctypes.c_void_p]
        lib.cudaFree.restype = ctypes.c_int
        lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.cudaMemcpyAsync.restype = ctypes.c_int

    def _check(self, code: int, operation: str) -> None:
        if code == 0:
            return
        message = self._lib.cudaGetErrorString(code)
        detail = message.decode("utf-8", errors="replace") if message else str(code)
        raise TensorRTError(f"{operation} failed with CUDA error {code}: {detail}")

    def _set_device(self) -> None:
        self._check(self._lib.cudaSetDevice(self._device_id), "cudaSetDevice")

    @property
    def stream_handle(self) -> int:
        return int(self._stream.value or 0)

    def malloc(self, size: int) -> int:
        self._set_device()
        pointer = ctypes.c_void_p()
        self._check(
            self._lib.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(int(size))),
            "cudaMalloc",
        )
        if not pointer.value:
            raise TensorRTError("cudaMalloc returned a null device pointer")
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        if not pointer:
            return
        self._set_device()
        self._check(self._lib.cudaFree(ctypes.c_void_p(pointer)), "cudaFree")

    def copy_host_to_device(self, pointer: int, array: np.ndarray) -> None:
        self._set_device()
        self._check(
            self._lib.cudaMemcpyAsync(
                ctypes.c_void_p(pointer),
                ctypes.c_void_p(array.ctypes.data),
                ctypes.c_size_t(array.nbytes),
                self._MEMCPY_HOST_TO_DEVICE,
                self._stream,
            ),
            "cudaMemcpyAsync(H2D)",
        )

    def copy_device_to_host(self, array: np.ndarray, pointer: int) -> None:
        self._set_device()
        self._check(
            self._lib.cudaMemcpyAsync(
                ctypes.c_void_p(array.ctypes.data),
                ctypes.c_void_p(pointer),
                ctypes.c_size_t(array.nbytes),
                self._MEMCPY_DEVICE_TO_HOST,
                self._stream,
            ),
            "cudaMemcpyAsync(D2H)",
        )

    def synchronize(self) -> None:
        self._set_device()
        self._check(self._lib.cudaStreamSynchronize(self._stream), "cudaStreamSynchronize")

    def close(self) -> None:
        stream = getattr(self, "_stream", None)
        if stream is None or not stream.value:
            return
        self._stream = ctypes.c_void_p()
        self._set_device()
        self._check(self._lib.cudaStreamDestroy(stream), "cudaStreamDestroy")


# ── TensorRT engine ──────────────────────────────────────────────────────────

class TensorRTEngine:
    """One deserialized engine + execution context + reusable CUDA buffers.

    Supports exactly one input tensor (any number of outputs). Static and
    dynamic-shape engines are handled the same way: `infer()` selects the
    optimization profile covering the input shape, sets it, resolves output
    shapes, grows device buffers on demand and runs H2D → execute → D2H →
    sync on the engine's own stream. Calls are serialized with a lock, so an
    instance may be shared between threads.
    """

    def __init__(self, path: str | Path, *, device_id: int = 0):
        import tensorrt as trt

        self.path = str(path)
        self._trt = trt
        self._lock = threading.Lock()
        self._cuda: CudaRuntime | None = None
        self._runtime = None
        self._engine = None
        self._context = None
        self._device_buffers: dict[str, int] = {}
        self._device_capacity: dict[str, int] = {}
        self._active_profile = 0
        self._active_shape: tuple[int, ...] | None = None

        self.metadata, serialized = read_engine_file(self.path)
        try:
            self._cuda = CudaRuntime(device_id)
            self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self._engine = self._runtime.deserialize_cuda_engine(serialized)
            del serialized
            if self._engine is None:
                raise TensorRTError(f"failed to deserialize TensorRT engine: {self.path}")
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise TensorRTError(f"failed to create TensorRT context: {self.path}")

            names = [
                self._engine.get_tensor_name(index)
                for index in range(self._engine.num_io_tensors)
            ]
            inputs = [
                name for name in names
                if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            ]
            if len(inputs) != 1:
                raise TensorRTError(
                    f"TensorRT engine must have exactly one input; got {inputs} ({self.path})"
                )
            self.input_name: str = inputs[0]
            self.output_names: list[str] = [name for name in names if name not in inputs]
            if not self.output_names:
                raise TensorRTError(f"TensorRT engine has no outputs ({self.path})")
            self.input_dtype: np.dtype = trt_dtype_to_numpy(
                trt, self._engine.get_tensor_dtype(self.input_name)
            )
            self.output_dtypes: dict[str, np.dtype] = {
                name: trt_dtype_to_numpy(trt, self._engine.get_tensor_dtype(name))
                for name in self.output_names
            }
            self.profiles: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = (
                self._read_profiles()
            )
        except Exception:
            self.close()
            raise

    # ── shape / profile helpers ──────────────────────────────────────────

    def _read_profiles(self):
        static_shape = tuple(int(v) for v in self._engine.get_tensor_shape(self.input_name))
        if static_shape and all(v > 0 for v in static_shape):
            return [(static_shape, static_shape, static_shape)]
        profiles = []
        for index in range(self._engine.num_optimization_profiles):
            minimum, optimum, maximum = self._engine.get_tensor_profile_shape(
                self.input_name, index
            )
            profiles.append(
                tuple(tuple(int(v) for v in shape) for shape in (minimum, optimum, maximum))
            )
        if not profiles:
            raise TensorRTError(f"TensorRT engine has no optimization profile ({self.path})")
        return profiles

    @property
    def is_static(self) -> bool:
        return len(self.profiles) == 1 and self.profiles[0][0] == self.profiles[0][2]

    @property
    def input_shape(self) -> tuple[int, ...] | None:
        """Static input shape, or None for dynamic engines."""
        return self.profiles[0][0] if self.is_static else None

    @property
    def optimization_shape(self) -> tuple[int, ...]:
        """Optimum shape of the first profile (useful for warm-up)."""
        return self.profiles[0][1]

    def select_profile(self, shape: tuple[int, ...]) -> int:
        """Return the profile index best covering `shape` (TensorRTShapeError)."""
        candidates = []
        for index, (minimum, optimum, maximum) in enumerate(self.profiles):
            if len(shape) != len(minimum):
                continue
            if all(lo <= v <= hi for v, lo, hi in zip(shape, minimum, maximum)):
                distance = sum(
                    abs(math.log(max(1, v) / max(1, o))) for v, o in zip(shape, optimum)
                )
                candidates.append((distance, index))
        if candidates:
            return min(candidates)[1]
        ranges = [{"min": lo, "max": hi} for lo, _opt, hi in self.profiles]
        raise TensorRTShapeError(
            f"TensorRT input shape {tuple(shape)} is outside profiles {ranges} ({self.path})"
        )

    # ── execution ────────────────────────────────────────────────────────

    def infer(self, array: np.ndarray) -> list[np.ndarray]:
        """Run the engine on one input array; returns outputs in output_names order."""
        array = np.ascontiguousarray(array, dtype=self.input_dtype)
        shape = tuple(int(v) for v in array.shape)
        with self._lock:
            if self._context is None or self._cuda is None:
                raise TensorRTError(f"TensorRT engine is closed ({self.path})")
            cuda = self._cuda
            context = self._context
            if shape != self._active_shape:
                profile = self.select_profile(shape)
                if profile != self._active_profile:
                    if not context.set_optimization_profile_async(profile, cuda.stream_handle):
                        raise TensorRTError(f"failed to select TensorRT profile {profile}")
                    self._active_profile = profile
                if not context.set_input_shape(self.input_name, shape):
                    raise TensorRTError(f"TensorRT rejected input shape {shape} ({self.path})")
                self._active_shape = shape

            outputs: list[np.ndarray] = []
            for name in self.output_names:
                output_shape = tuple(int(v) for v in context.get_tensor_shape(name))
                if any(v < 0 for v in output_shape):
                    raise TensorRTError(
                        f"TensorRT produced unresolved output shape {output_shape} for {name}"
                    )
                outputs.append(np.empty(output_shape, dtype=self.output_dtypes[name]))

            input_pointer = self._ensure_capacity(self.input_name, array.nbytes)
            context.set_tensor_address(self.input_name, input_pointer)
            for name, output in zip(self.output_names, outputs):
                context.set_tensor_address(name, self._ensure_capacity(name, output.nbytes))

            cuda.copy_host_to_device(input_pointer, array)
            if not context.execute_async_v3(cuda.stream_handle):
                raise TensorRTError(f"TensorRT execute_async_v3 failed ({self.path})")
            for name, output in zip(self.output_names, outputs):
                cuda.copy_device_to_host(output, self._device_buffers[name])
            cuda.synchronize()
            return outputs

    def _ensure_capacity(self, name: str, required: int) -> int:
        required = max(1, int(required))
        pointer = self._device_buffers.get(name, 0)
        if pointer and required <= self._device_capacity.get(name, 0):
            return pointer
        if pointer:
            self._cuda.free(pointer)
            self._device_buffers[name] = 0
            self._device_capacity[name] = 0
        pointer = self._cuda.malloc(required)
        self._device_buffers[name] = pointer
        self._device_capacity[name] = required
        return pointer

    # ── lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        # Taken under the same lock as infer(): a model-affecting config can
        # close a stale adapter while a worker that missed its stop deadline
        # is still inside infer(). Without the lock that frees CUDA buffers
        # and destroys the stream under an in-flight execution — a
        # use-after-free. With it, close() blocks until the in-flight infer()
        # finishes, and any later infer() sees _context is None and raises.
        lock = getattr(self, "_lock", None)
        if lock is None:  # partially constructed instance (__del__ path)
            return
        with lock:
            cuda = getattr(self, "_cuda", None)
            buffers = getattr(self, "_device_buffers", {})
            if cuda is not None:
                for name, pointer in list(buffers.items()):
                    try:
                        cuda.free(pointer)
                    except Exception:
                        log.debug("cudaFree failed for %s", name, exc_info=True)
            self._device_buffers = {}
            self._device_capacity = {}
            self._active_shape = None
            self._context = None
            self._engine = None
            self._runtime = None
            if cuda is not None:
                self._cuda = None
                try:
                    cuda.close()
                except Exception:
                    log.debug("cudaStreamDestroy failed", exc_info=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Interpreter shutdown can unload CUDA before Python finalizers run.
            pass
