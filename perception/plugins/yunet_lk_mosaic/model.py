"""Pinned YuNet model and OpenCV / shared Perception ORT backends."""
from __future__ import annotations

import os
import uuid

import numpy as np

from utils.model_downloader import (
    _verify_pinned_file, ensure_verified_bundle, require_models_subpath,
)

from .config import MosaicError

MODEL_FILE = "face_detection_yunet_2023mar.onnx"
MODEL_METADATA = {
    "size": 232589,
    "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
}
MODEL_SOURCES = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    "47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/"
    "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310",
)


def resolve_model(path, progress=None):
    if path is not None:
        try:
            _verify_pinned_file(path, MODEL_METADATA)
        except (ValueError, OSError) as exc:
            raise MosaicError(f"model_verification_failed: {exc}") from exc
        return path
    model_dir = require_models_subpath("/models/yunet_lk_mosaic")
    sources = os.environ.get("YUNET_LK_MOSAIC_MODEL_BASE_URL") or MODEL_SOURCES
    return ensure_verified_bundle(
        "yunet_lk_mosaic", model_dir, sources, {MODEL_FILE: MODEL_METADATA},
        progress_cb=progress,
    )[MODEL_FILE]


class YuNetDetector:
    def __init__(self, cv, model_path, config):
        self.cv = cv
        self.path = model_path
        self.config = config
        self.native = None
        self.service = None
        self.backend = "opencv"
        create = getattr(getattr(cv, "FaceDetectorYN", None), "create", None)
        if callable(create):
            try:
                self.native = create(model_path, "", (320, 320),
                                     config.confidence_threshold, 0.3, 5000)
            except cv.error:
                pass
        if self.native is None:
            self._start_ort()

    def _start_ort(self):
        from plugins.ort_worker import get_worker

        worker = get_worker()
        key = "yunet_lk_mosaic." + uuid.uuid4().hex
        try:
            self.service = worker.service(
                key, "plugins.yunet_lk_mosaic.yunet_ort", "YuNetOrtService",
                threshold=self.config.confidence_threshold, limit=self.config.max_faces + 1,
                _ort_session={
                    "model_path": self.path,
                    "providers": ["CPUExecutionProvider"],
                    "options": {
                        "intra_op_num_threads": 2,
                        "session_config": {"session.intra_op.allow_spinning": "0"},
                    },
                },
            )
        except Exception:
            worker.drop(key)
            raise
        self.backend = "onnxruntime-cpu"
        self.native = None

    def detect(self, image):
        height, width = image.shape[:2]
        if self.native is not None:
            try:
                self.native.setInputSize((width, height))
                return self.native.detect(image)[1]
            except self.cv.error:
                self._start_ort()
        limit = self.service.describe()["max_side"]
        scale = min(1.0, limit / max(width, height))
        sw, sh = max(1, round(width * scale)), max(1, round(height * scale))
        small = image if scale == 1 else self.cv.resize(
            image, (sw, sh), interpolation=self.cv.INTER_AREA,
        )
        rows = self.service.call("detect", image=np.ascontiguousarray(small))
        rows[:, :4] *= [width / sw, height / sh, width / sw, height / sh]
        return rows

    def close(self):
        self.native = None
        if self.service is not None:
            service, self.service = self.service, None
            service.drop()
