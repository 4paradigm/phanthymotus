"""Direct TensorRT runner using cuda-python and NumPy host arrays."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def _load_cuda_runtime():
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart
    return cudart


def _check_cuda(result, operation: str):
    if not isinstance(result, tuple):
        result = (result,)
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"CUDA {operation} failed with error {int(error)}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CudaRuntime:
    """One CUDA stream shared by all three sequential TensorRT engines."""

    def __init__(self):
        self.api = _load_cuda_runtime()
        self.stream = _check_cuda(self.api.cudaStreamCreate(), "stream creation")
        self._closed = False

    def malloc(self, size: int) -> int:
        return int(_check_cuda(self.api.cudaMalloc(size), "allocation"))

    def free(self, pointer: int) -> None:
        if pointer:
            _check_cuda(self.api.cudaFree(pointer), "free")

    def copy_to_device(self, pointer: int, array: np.ndarray) -> None:
        kind = self.api.cudaMemcpyKind.cudaMemcpyHostToDevice
        _check_cuda(
            self.api.cudaMemcpyAsync(
                pointer, int(array.ctypes.data), array.nbytes, kind, self.stream
            ),
            "host-to-device copy",
        )

    def copy_to_host(self, array: np.ndarray, pointer: int) -> None:
        kind = self.api.cudaMemcpyKind.cudaMemcpyDeviceToHost
        _check_cuda(
            self.api.cudaMemcpyAsync(
                int(array.ctypes.data), pointer, array.nbytes, kind, self.stream
            ),
            "device-to-host copy",
        )

    def synchronize(self) -> None:
        _check_cuda(self.api.cudaStreamSynchronize(self.stream), "stream synchronize")

    def compute_capability(self) -> str:
        properties = _check_cuda(self.api.cudaGetDeviceProperties(0), "device query")
        return f"{int(properties.major)}.{int(properties.minor)}"

    def close(self) -> None:
        if not self._closed:
            _check_cuda(self.api.cudaStreamDestroy(self.stream), "stream destruction")
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _DeviceBuffer:
    def __init__(self, cuda: CudaRuntime):
        self.cuda = cuda
        self.pointer = 0
        self.capacity = 0

    def reserve(self, size: int) -> int:
        if size > self.capacity:
            self.cuda.free(self.pointer)
            self.pointer = self.cuda.malloc(size)
            self.capacity = size
        return self.pointer

    def close(self) -> None:
        self.cuda.free(self.pointer)
        self.pointer = 0
        self.capacity = 0


class TensorRTCudaSession:
    def __init__(
        self,
        engine_path: str | Path,
        cuda: CudaRuntime,
        expected_sha256: str | None = None,
    ):
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError("TensorRT Python bindings are required") from exc

        self.trt = trt
        self.cuda = cuda
        self.path = Path(engine_path)
        if expected_sha256 and _sha256(self.path) != expected_sha256:
            raise RuntimeError(f"TensorRT engine checksum mismatch: {self.path}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT context: {self.path}")
        if not hasattr(self.engine, "num_io_tensors"):
            raise RuntimeError("The CUDA/NumPy backend requires TensorRT tensor API v3")
        self._buffers: dict[str, _DeviceBuffer] = {}

    def _buffer(self, name: str, size: int) -> int:
        return self._buffers.setdefault(name, _DeviceBuffer(self.cuda)).reserve(size)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        trt = self.trt
        host_inputs = {}
        output_specs = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            if mode == trt.TensorIOMode.INPUT:
                if name not in inputs:
                    raise KeyError(f"Missing TensorRT input {name!r}")
                array = np.ascontiguousarray(inputs[name], dtype=dtype)
                if not self.context.set_input_shape(name, array.shape):
                    raise ValueError(f"Shape outside TensorRT profile: {name}={array.shape}")
                host_inputs[name] = array
            else:
                output_specs.append((name, dtype))

        outputs = {}
        for name, array in host_inputs.items():
            pointer = self._buffer(name, array.nbytes)
            self.cuda.copy_to_device(pointer, array)
            self.context.set_tensor_address(name, pointer)
        for name, dtype in output_specs:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"Unresolved TensorRT output shape: {name}={shape}")
            array = np.empty(shape, dtype=dtype)
            pointer = self._buffer(name, array.nbytes)
            self.context.set_tensor_address(name, pointer)
            outputs[name] = array

        if not self.context.execute_async_v3(self.cuda.stream):
            raise RuntimeError(f"TensorRT execution failed: {self.path.name}")
        for name, array in outputs.items():
            self.cuda.copy_to_host(array, self._buffers[name].pointer)
        self.cuda.synchronize()
        return outputs

    def close(self) -> None:
        for buffer in self._buffers.values():
            buffer.close()
        self._buffers.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
