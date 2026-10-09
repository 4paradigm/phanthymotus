"""K-aware metric depth using a prebuilt TensorRT engine and SelectiveScan plugin."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


def camera_matrix(camera):
    """Read a motus.camera/1 declaration; K is expressed at width × height."""
    if not isinstance(camera, dict):
        raise ValueError("DepthART requires camera_info with K, width and height")
    try:
        matrix = np.asarray(camera["K"], dtype=np.float32)
        width, height = int(camera["width"]), int(camera["height"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("DepthART requires camera_info with a nine-value K, width and height") from exc
    if matrix.shape != (9,) or width <= 0 or height <= 0:
        raise ValueError("camera_info must contain a nine-value K and positive width/height")
    matrix = matrix.reshape(3, 3).copy()
    if (not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-6)
            or abs(float(matrix[1, 0])) > 1e-6):
        raise ValueError("camera_info K must be finite pinhole intrinsics with positive focal lengths")
    return matrix, width, height


def prepare_image(frame, camera, input_size):
    """Official lower-bound/multiple-of-32 resize, with matching K and RGB normalization."""
    import cv2

    matrix, camera_w, camera_h = camera_matrix(camera)
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0:
        raise ValueError("DepthART requires a nonempty BGR image")
    height, width = frame.shape[:2]
    target_w, target_h = input_size
    scale = max(target_w / width, target_h / height)
    def rounded(value, minimum):
        size = int(np.round(value / 32) * 32)
        return size if size >= minimum else int(np.ceil(value / 32) * 32)
    new_w, new_h = rounded(width * scale, target_w), rounded(height * scale, target_h)
    if (new_w, new_h) != (target_w, target_h):
        raise ValueError(
            f"DepthART engine canvas {target_w}x{target_h} does not support this image aspect ratio "
            f"({width}x{height}); its aspect-preserving resize is {new_w}x{new_h}"
        )
    # The declaration may describe a higher-resolution version of the same full frame.
    matrix[0] *= new_w / camera_w
    matrix[1] *= new_h / camera_h
    image = frame[:, :, ::-1].astype(np.float32) / 255.0
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    image = (image - np.asarray([.485, .456, .406], dtype=np.float32)) / np.asarray([.229, .224, .225], dtype=np.float32)
    blob = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
    return blob, matrix[None]


class _CameraSession:
    def __init__(self, session, camera):
        self._session, self._camera = session, camera

    def infer(self, frame):
        camera = self._camera() if callable(self._camera) else self._camera
        return self._session.infer(frame, camera)


class DepthARTSession:
    def __init__(self, engine_path, plugin_path):
        from utils.tensorrt_runtime import TensorRTEngine

        if not engine_path or not plugin_path:
            raise ValueError("DepthART requires provisioned depthart_engine_path and depthart_plugin_path")
        engine_path, plugin_path = Path(engine_path), Path(plugin_path)
        if not engine_path.is_file() or not plugin_path.is_file():
            raise FileNotFoundError("DepthART engine/plugin missing; provision artifacts matching this JetPack/TensorRT")
        self._plugin = ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
        try:
            version = self._plugin.depthart_selective_scan_trt_version
        except AttributeError as exc:
            raise ValueError("DepthART plugin is missing its version symbol; provision a matching SelectiveScan plugin") from exc
        version.restype = ctypes.c_char_p
        if version() != b"SelectiveScan-1":
            raise ValueError("unsupported DepthART SelectiveScan plugin version")
        self._engine = TensorRTEngine(engine_path, primary_input="image")
        shape = self._engine.input_shape
        if (shape is None or len(shape) != 4 or shape[:2] != (1, 3)
                or self._engine.auxiliary_shapes != {"camera_K": (1, 3, 3)}):
            self._engine.close()
            raise ValueError("DepthART requires static image [1,3,H,W] and camera_K [1,3,3] inputs")
        self.input_size = (shape[3], shape[2])

    def for_camera(self, camera):
        return _CameraSession(self, camera)

    def infer(self, frame, camera):
        blob, matrix = prepare_image(frame, camera, self.input_size)
        return self._engine.infer(blob, auxiliary_inputs={"camera_K": matrix}), None

    def close(self):
        self._engine.close()
