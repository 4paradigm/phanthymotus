"""Small direct TensorRT runner backed by Torch CUDA allocations."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

_TRT_LOGGER = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_dtype(trt, dtype):
    numpy_dtype = np.dtype(trt.nptype(dtype))
    mapping = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.bool_): torch.bool,
    }
    if numpy_dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[numpy_dtype]


class TensorRTSession:
    def __init__(self, engine_path: str | Path, expected_sha256: str | None = None):
        try:
            import tensorrt as trt
        except ImportError as exc:  # pragma: no cover - Jetson dependency
            raise RuntimeError("TensorRT Python bindings are required") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT inference requires a CUDA device")

        self.trt = trt
        self.path = Path(engine_path)
        if expected_sha256 and _sha256(self.path) != expected_sha256:
            raise RuntimeError(f"TensorRT engine checksum mismatch: {self.path}")
        global _TRT_LOGGER
        if _TRT_LOGGER is None:
            _TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.logger = _TRT_LOGGER
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT context: {self.path}")
        self.v3_api = hasattr(self.engine, "num_io_tensors")
        self.stream = torch.cuda.Stream()

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.v3_api:
            return self._run_v3(inputs)
        return self._run_v2(inputs)

    def _run_v3(self, inputs):
        trt = self.trt
        tensors = {}
        output_names = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            dtype = _torch_dtype(trt, self.engine.get_tensor_dtype(name))
            if mode == trt.TensorIOMode.INPUT:
                if name not in inputs:
                    raise KeyError(f"Missing TensorRT input {name!r}")
                tensor = inputs[name].to(device="cuda", dtype=dtype).contiguous()
                if not self.context.set_input_shape(name, tuple(tensor.shape)):
                    raise ValueError(f"Shape outside TensorRT profile: {name}={tuple(tensor.shape)}")
                tensors[name] = tensor
            else:
                output_names.append(name)
        for name in output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"Unresolved TensorRT output shape: {name}={shape}")
            tensors[name] = torch.empty(
                shape, device="cuda", dtype=_torch_dtype(trt, self.engine.get_tensor_dtype(name))
            )
        for name, tensor in tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        current_stream = torch.cuda.current_stream()
        self.stream.wait_stream(current_stream)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution failed: {self.path.name}")
        current_stream.wait_stream(self.stream)
        return {name: tensors[name] for name in output_names}

    def _run_v2(self, inputs):
        trt = self.trt
        bindings = [0] * self.engine.num_bindings
        tensors = {}
        output_names = []
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            dtype = _torch_dtype(trt, self.engine.get_binding_dtype(index))
            if self.engine.binding_is_input(index):
                if name not in inputs:
                    raise KeyError(f"Missing TensorRT input {name!r}")
                tensor = inputs[name].to(device="cuda", dtype=dtype).contiguous()
                if not self.context.set_binding_shape(index, tuple(tensor.shape)):
                    raise ValueError(f"Shape outside TensorRT profile: {name}={tuple(tensor.shape)}")
                tensors[name] = tensor
                bindings[index] = tensor.data_ptr()
            else:
                output_names.append(name)
        if not self.context.all_binding_shapes_specified:
            raise RuntimeError("Not all TensorRT binding shapes were specified")
        for name in output_names:
            index = self.engine.get_binding_index(name)
            shape = tuple(self.context.get_binding_shape(index))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"Unresolved TensorRT output shape: {name}={shape}")
            tensor = torch.empty(shape, device="cuda", dtype=_torch_dtype(trt, self.engine.get_binding_dtype(index)))
            tensors[name] = tensor
            bindings[index] = tensor.data_ptr()
        current_stream = torch.cuda.current_stream()
        self.stream.wait_stream(current_stream)
        if not self.context.execute_async_v2(bindings, self.stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution failed: {self.path.name}")
        current_stream.wait_stream(self.stream)
        return {name: tensors[name] for name in output_names}
