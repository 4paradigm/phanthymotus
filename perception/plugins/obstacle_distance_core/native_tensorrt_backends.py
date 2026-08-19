"""Obstacle TensorRT backends (YOLO seg / depth) on the shared TensorRT runtime.

Engine deserialization, dtype mapping, CUDA buffers/stream and execution live
in utils.tensorrt_runtime.TensorRTEngine (shared with the OCR plugin); this
module only adds the fixed-shape contract, Ultralytics metadata checks and
the YOLO/ZipDepth pre/post-processing.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Sequence

import numpy as np

from utils.tensorrt_runtime import TensorRTEngine, TensorRTError

from .contracts import (
    ErrorCode,
    InstanceMask,
    ObstacleDistanceError,
)
from .runtime_utils import (
    SEG_CONF_FLOOR,
    SEG_INPUT_SIZE,
    check_deadline,
    decode_image,
    letterbox,
)


log = logging.getLogger(__name__)

_YOLO_DEPTH_INPUT_SIZE = 768


class _NativeTensorRTEngine:
    """Fixed-shape view over a shared TensorRTEngine with obstacle error mapping.

    Obstacle engines are exported with static input/output shapes; anything
    dynamic is rejected up front so the estimator reports a structured
    model_error instead of failing on the first frame.
    """

    def __init__(self, path: str, expected_task: str | None = None) -> None:
        try:
            self._engine = TensorRTEngine(path)
        except TensorRTError as error:
            raise ObstacleDistanceError(ErrorCode.MODEL_ERROR, str(error)) from None
        try:
            self.metadata = self._engine.metadata
            if expected_task and self.metadata.get("task") != expected_task:
                raise ObstacleDistanceError(
                    ErrorCode.MODEL_ERROR,
                    f"TensorRT engine task must be {expected_task}",
                )
            if not self._engine.is_static:
                raise ObstacleDistanceError(
                    ErrorCode.MODEL_ERROR,
                    "native TensorRT backend requires fixed input shapes",
                )
            self.input_name = self._engine.input_name
            self.input_shape = tuple(self._engine.input_shape)
            self.input_dtype = self._engine.input_dtype
            self.output_names = list(self._engine.output_names)
            self.output_dtypes = dict(self._engine.output_dtypes)
        except Exception:
            self._engine.close()
            raise

    def infer(self, image: np.ndarray) -> tuple[np.ndarray, ...]:
        image = np.ascontiguousarray(image, dtype=self.input_dtype)
        if tuple(image.shape) != self.input_shape:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                f"TensorRT input shape mismatch: {image.shape} != {self.input_shape}",
            )
        try:
            return tuple(self._engine.infer(image))
        except TensorRTError as error:
            raise ObstacleDistanceError(ErrorCode.MODEL_ERROR, str(error)) from None

    def close(self) -> None:
        engine = getattr(self, "_engine", None)
        if engine is not None:
            engine.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _prepare_yolo_image(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    letterboxed, ratio, dw, dh = letterbox(image, size)
    rgb = letterboxed[:, :, ::-1]
    chw = np.transpose(rgb, (2, 0, 1))
    tensor = np.ascontiguousarray(chw, dtype=np.float32)[None] / 255.0
    return tensor, ratio, dw, dh


def _resize_align_corners(image: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    if image.shape == (height, width):
        return image.astype(np.float32, copy=False)
    source_height, source_width = image.shape
    x = np.linspace(0, source_width - 1, width, dtype=np.float32)
    y = np.linspace(0, source_height - 1, height, dtype=np.float32)
    map_x, map_y = np.meshgrid(x, y)
    return cv2.remap(
        image.astype(np.float32, copy=False),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _scale_depth_to_original(
    depth: np.ndarray,
    original_height: int,
    original_width: int,
) -> np.ndarray:
    import cv2

    input_height, input_width = depth.shape
    gain = min(input_height / original_height, input_width / original_width)
    pad_w = (input_width - round(original_width * gain)) / 2
    pad_h = (input_height - round(original_height * gain)) / 2
    top = round(pad_h - 0.1)
    left = round(pad_w - 0.1)
    bottom = input_height - round(pad_h + 0.1)
    right = input_width - round(pad_w + 0.1)
    cropped = depth[top:bottom, left:right]
    return cv2.resize(
        cropped.astype(np.float32, copy=False),
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )


def _process_yolo_masks(
    detections: np.ndarray,
    prototypes: np.ndarray,
    original_height: int,
    original_width: int,
    ratio: float,
    dw: int,
    dh: int,
    *,
    confidence_floor: float = SEG_CONF_FLOOR,
    allowed_class_ids: frozenset[int] | None = None,
) -> list[tuple[int, float, np.ndarray]]:
    import cv2

    detections = np.asarray(detections, dtype=np.float32)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    if detections.ndim == 3:
        detections = detections[0]
    if prototypes.ndim == 4:
        prototypes = prototypes[0]
    selected_mask = detections[:, 4] > SEG_CONF_FLOOR
    selected_mask &= detections[:, 4] >= confidence_floor
    if allowed_class_ids is not None:
        selected_mask &= np.isin(
            detections[:, 5].astype(np.int64),
            tuple(allowed_class_ids),
        )
    selected = detections[selected_mask]
    if not len(selected):
        return []
    channels, mask_height, mask_width = prototypes.shape
    coefficients = selected[:, 6 : 6 + channels]
    logits = (coefficients @ prototypes.reshape(channels, -1)).reshape(
        -1,
        mask_height,
        mask_width,
    )
    unpad_height = min(
        int(round(original_height * ratio)),
        SEG_INPUT_SIZE - dh,
    )
    unpad_width = min(
        int(round(original_width * ratio)),
        SEG_INPUT_SIZE - dw,
    )
    rows = np.arange(SEG_INPUT_SIZE, dtype=np.float32)[:, None]
    columns = np.arange(SEG_INPUT_SIZE, dtype=np.float32)[None, :]
    results = []
    for detection, logit in zip(selected, logits):
        upsampled = cv2.resize(
            logit,
            (SEG_INPUT_SIZE, SEG_INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        x1, y1, x2, y2 = detection[:4]
        mask = upsampled > 0.0
        mask &= columns >= x1
        mask &= columns < x2
        mask &= rows >= y1
        mask &= rows < y2
        if not mask.any():
            continue
        mask = mask[dh : dh + unpad_height, dw : dw + unpad_width]
        mask = cv2.resize(
            mask.astype(np.uint8),
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        results.append((int(detection[5]), float(detection[4]), mask))
    return results


class NativeTensorRTSegBackend:
    def __init__(
        self,
        engine: str,
        *,
        allowed_classes: object = None,
        min_confidence: object = None,
    ) -> None:
        self._engine_path = engine
        self._allowed_classes = allowed_classes
        self._names: dict[int, str] = {}
        self._allowed_class_ids: frozenset[int] | None = None
        self._confidence_floor = self._configured_confidence(min_confidence)
        # Load the engine on the first vehicle frame so an indoor-only
        # deployment does not pay for segmentation residency.
        self._model: _NativeTensorRTEngine | None = None
        self._engine_init_lock = threading.Lock()
        log.info("[obstacle] native TensorRT segmentation configured %s", engine)

    def _get_model(self) -> _NativeTensorRTEngine:
        model = self._model
        if model is None:
            with self._engine_init_lock:
                if self._model is None:
                    started = time.monotonic()
                    model = _NativeTensorRTEngine(
                        self._engine_path,
                        expected_task="segment",
                    )
                    if model.input_shape != (
                        1,
                        3,
                        SEG_INPUT_SIZE,
                        SEG_INPUT_SIZE,
                    ):
                        raise ObstacleDistanceError(
                            ErrorCode.MODEL_ERROR,
                            "YOLO segmentation TensorRT engine input shape is incompatible",
                        )
                    names = model.metadata.get("names", {})
                    if not isinstance(names, dict):
                        raise ObstacleDistanceError(
                            ErrorCode.MODEL_ERROR,
                            "YOLO segmentation metadata does not contain class names",
                        )
                    self._names = {
                        int(key): str(value) for key, value in names.items()
                    }
                    self._allowed_class_ids = self._configured_class_ids(
                        self._allowed_classes
                    )
                    self._model = model
                    log.info(
                        "[obstacle] native TensorRT segmentation loaded %s "
                        "in %.1fms",
                        self._engine_path,
                        1000.0 * (time.monotonic() - started),
                    )
                model = self._model
        return model

    def _configured_class_ids(
        self,
        allowed_classes: object,
    ) -> frozenset[int] | None:
        if not isinstance(allowed_classes, (list, tuple, set, frozenset)):
            return None
        if not allowed_classes or any(
            not isinstance(value, str) for value in allowed_classes
        ):
            return None
        allowed = set(allowed_classes)
        return frozenset(
            class_id
            for class_id, class_name in self._names.items()
            if class_name in allowed
        )

    @staticmethod
    def _configured_confidence(min_confidence: object) -> float:
        if (
            isinstance(min_confidence, bool)
            or not isinstance(min_confidence, (int, float))
        ):
            return SEG_CONF_FLOOR
        confidence = float(min_confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return SEG_CONF_FLOOR
        return max(SEG_CONF_FLOOR, confidence)

    def close(self) -> None:
        """Release the segmentation engine and its CUDA buffers."""
        with self._engine_init_lock:
            model = self._model
            self._model = None
        if model is not None:
            model.close()

    def predict_instances(
        self,
        image_bytes: bytes,
        deadline_monotonic: float,
    ) -> Sequence[InstanceMask]:
        check_deadline(deadline_monotonic)
        image = decode_image(image_bytes)
        height, width = image.shape[:2]
        tensor, ratio, dw, dh = _prepare_yolo_image(image, SEG_INPUT_SIZE)
        model = self._get_model()
        outputs = model.infer(tensor)
        detection_outputs = [output for output in outputs if output.ndim == 3]
        prototype_outputs = [output for output in outputs if output.ndim == 4]
        if len(detection_outputs) != 1 or len(prototype_outputs) != 1:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "YOLO segmentation engine outputs are incompatible",
            )
        processed = _process_yolo_masks(
            detection_outputs[0],
            prototype_outputs[0],
            height,
            width,
            ratio,
            dw,
            dh,
            confidence_floor=self._confidence_floor,
            allowed_class_ids=self._allowed_class_ids,
        )
        check_deadline(deadline_monotonic)
        return tuple(
            InstanceMask(
                class_name=self._names.get(class_id, str(class_id)),
                confidence=confidence,
                mask=mask,
            )
            for class_id, confidence, mask in processed
        )
