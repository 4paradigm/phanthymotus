"""Standalone, *weak-privacy* smooth-mosaic processing (no ROS or HTTP).

This is a demo/visual effect, NOT a privacy guarantee. Non-detected faces may
remain recognizable in local-mosaic mode. The caller must obtain consent.
Every output is generated from a decoded image and freshly re-encoded; never
return original JPEG bytes or an unredacted image on failures.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
import uuid
from dataclasses import dataclass, replace

from plugins.yunet_ort import cpu_session_spec
from utils.model_downloader import (
    MOSAIC_SMOOTH_MODEL_FILENAME, MOSAIC_SMOOTH_MODEL_FILES,
)

MAX_JPEG_BYTES = 8 * 1024 * 1024
MAX_SOURCE_SIDE = 2048
COAST_LIMIT_S = 0.55


class MosaicDemoError(ValueError):
    pass


@dataclass(frozen=True)
class DemoConfig:
    output_width: int = 640
    detect_every_frames: int = 4
    confidence_threshold: float = 0.28
    max_faces: int = 6
    jpeg_quality: int = 72
    pixelate_block: int = 24
    fallback_block: int = 40
    consent_acknowledged: bool = False

    def updated(self, values: dict) -> "DemoConfig":
        if not isinstance(values, dict):
            raise MosaicDemoError("config_must_be_object")
        extra = set(values) - set(self.__dataclass_fields__)
        if extra:
            raise MosaicDemoError("unknown_config_field:" + sorted(extra)[0])
        cfg = replace(self, **values)
        for name, low, high in (
            ("output_width", 320, 960),
            ("detect_every_frames", 1, 12),
            ("max_faces", 1, 8),
            ("jpeg_quality", 35, 95),
            ("pixelate_block", 8, 64),
            ("fallback_block", 16, 96),
        ):
            v = getattr(cfg, name)
            if type(v) is not int or not low <= v <= high:
                raise MosaicDemoError("invalid_config:" + name)
        if (type(cfg.confidence_threshold) not in (int, float)
                or not math.isfinite(cfg.confidence_threshold)
                or not 0.1 <= cfg.confidence_threshold <= 0.9):
            raise MosaicDemoError("invalid_config:confidence_threshold")
        if type(cfg.consent_acknowledged) is not bool:
            raise MosaicDemoError("invalid_config:consent_acknowledged")
        return cfg


def validate_model(path: str) -> str:
    if not isinstance(path, str) or not path:
        raise MosaicDemoError("model_path_required")
    if not os.path.isfile(path):
        raise MosaicDemoError("model_missing")
    metadata = MOSAIC_SMOOTH_MODEL_FILES[MOSAIC_SMOOTH_MODEL_FILENAME]
    if os.path.getsize(path) != metadata["size"]:
        raise MosaicDemoError("model_size_mismatch")
    digest_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest_hash.update(chunk)
    digest = digest_hash.hexdigest()
    if digest != metadata["sha256"]:
        raise MosaicDemoError("model_sha256_mismatch")
    return path


def _whole_mosaic(cv, image, block):
    h, w = image.shape[:2]
    small = cv.resize(
        image, (max(1, w // block), max(1, h // block)),
        interpolation=cv.INTER_AREA,
    )
    return cv.resize(small, (w, h), interpolation=cv.INTER_NEAREST)


def _mosaic_face(cv, image, box, block):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = (x2-x1)*0.35, (y2-y1)*0.35
    a, b, c, d = (
        max(0, int(x1-dx)), max(0, int(y1-dy)),
        min(w, int(x2+dx)), min(h, int(y2+dy)),
    )
    if c <= a or d <= b:
        return
    roi = image[b:d, a:c]
    sh, sw = roi.shape[:2]
    small = cv.resize(
        roi, (max(1, sw // block), max(1, sh // block)),
        interpolation=cv.INTER_AREA,
    )
    image[b:d, a:c] = cv.resize(
        small, (sw, sh), interpolation=cv.INTER_NEAREST,
    )


class SmoothMosaic:
    """One instance's algorithm state. All methods used by its single worker.

    Do NOT call cv2.setNumThreads here: that changes OpenCV process-wide and
    would perturb ASR/face/pose in the shared Perception container.
    """

    def __init__(self, model_path: str, config: DemoConfig, *, detector=None, cv=None):
        self.cfg = config
        if cv is None:
            import cv2 as cv
        self.cv = cv
        import numpy as np
        self.np = np
        self._model_path = self._ort_service = None
        self.detector = detector
        if detector is None:
            self._model_path = validate_model(model_path)
            create = getattr(getattr(cv, "FaceDetectorYN", None), "create", None)
            if callable(create):
                try:
                    self.detector = create(model_path, "", (320, 320), 0.22, 0.3, 5000)
                except cv.error:
                    pass
            if self.detector is None:
                self._start_ort()
        self.number = 0
        self.detect_calls = 0
        self.fallback_frames = 0
        self.last_boxes = []
        self.points = None
        self.prev_gray = None

    def _start_ort(self):
        from plugins.ort_worker import get_worker

        worker = get_worker()
        key = "mosaic_smooth_demo.yunet." + uuid.uuid4().hex
        try:
            self._ort_service = worker.service(
                key, "plugins.yunet_ort", "YuNetOrtService",
                threshold=self.cfg.confidence_threshold, limit=self.cfg.max_faces + 1,
                _ort_session=cpu_session_spec(self._model_path),
            )
        except Exception:
            worker.drop(key)
            raise
        self.detector = None
        self._input_limit = self._ort_service.describe()["max_side"]

    def _detect(self, image):
        height, width = image.shape[:2]
        if self.detector is not None:
            try:
                self.detector.setInputSize((width, height))
                return self.detector.detect(image)[1]
            except self.cv.error:
                if self._model_path is None:
                    raise
                self._start_ort()
        scale = min(1., self._input_limit / max(width, height))
        scaled_width = max(1, round(width * scale))
        scaled_height = max(1, round(height * scale))
        small = (image if scale == 1. else self.cv.resize(
            image, (scaled_width, scaled_height), interpolation=self.cv.INTER_AREA,
        ))
        rows = self._ort_service.call("detect", image=small)
        rows[:, :4] *= [width / scaled_width, height / scaled_height,
                         width / scaled_width, height / scaled_height]
        return rows

    def close(self):
        self.detector = None
        service, self._ort_service = self._ort_service, None
        if service is not None:
            service.drop()
        self.last_detect_ok = 0.0

    def process(self, jpeg: bytes):
        """Return (freshly encoded JPEG, detected/tracked face count, fallback).

        Failures raise: caller must publish *status only*, not source data.
        """
        if (not isinstance(jpeg, bytes) or not 4 <= len(jpeg) <= MAX_JPEG_BYTES
                or not jpeg.startswith(b"\xff\xd8")
                or not jpeg.endswith(b"\xff\xd9")):
            raise MosaicDemoError("invalid_jpeg")
        cv, np = self.cv, self.np
        raw = cv.imdecode(
            np.frombuffer(jpeg, np.uint8),
            cv.IMREAD_COLOR | cv.IMREAD_IGNORE_ORIENTATION,
        )
        if (raw is None or raw.ndim != 3 or raw.shape[2] != 3
                or min(raw.shape[:2]) < 1
                or max(raw.shape[:2]) > MAX_SOURCE_SIDE):
            raise MosaicDemoError("invalid_image")
        h, w = raw.shape[:2]
        if w != self.cfg.output_width:
            new_h = max(1, int(round(h*self.cfg.output_width/w)))
            if new_h > MAX_SOURCE_SIDE:
                raise MosaicDemoError("invalid_image_aspect_ratio")
            image = cv.resize(
                raw, (self.cfg.output_width, new_h),
                interpolation=cv.INTER_AREA,
            )
        else:
            image = raw
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        now = time.monotonic()
        self.number += 1
        do_detect = (self.number - 1) % self.cfg.detect_every_frames == 0
        if do_detect:
            self.detect_calls += 1
            faces = self._detect(image)
            boxes = []
            if faces is not None:
                for face in faces:
                    if float(face[-1]) < self.cfg.confidence_threshold:
                        continue
                    x, y, bw, bh = map(float, face[:4])
                    if (bw > 0 and bh > 0 and
                            all(math.isfinite(t) for t in (x,y,bw,bh))):
                        boxes.append((x, y, x+bw, y+bh))
                    else:
                        raise MosaicDemoError("invalid_detection")
            if 1 <= len(boxes) <= self.cfg.max_faces:
                self.last_boxes = boxes
                self.last_detect_ok = now
                mask = np.zeros_like(gray)
                for x1,y1,x2,y2 in boxes:
                    a,b = max(0,int(x1)), max(0,int(y1))
                    c,d = min(gray.shape[1],int(x2)),min(gray.shape[0],int(y2))
                    if c>a and d>b: mask[b:d,a:c]=255
                self.points = cv.goodFeaturesToTrack(
                    gray, maxCorners=75, qualityLevel=.025,
                    minDistance=6, mask=mask,
                )
            elif len(boxes) > self.cfg.max_faces:
                self.last_boxes = []
                self.points = None
            # Zero detection does NOT mean the scene has no people.
        elif (self.last_boxes and self.points is not None
              and self.prev_gray is not None):
            try:
                tracked, valid, _ = cv.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.points, None,
                    winSize=(15,15), maxLevel=2,
                    criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,10,.03),
                )
                if tracked is not None and valid is not None:
                    old = self.points.reshape(-1,2)
                    moved = tracked.reshape(-1,2)
                    accepted = valid.reshape(-1)==1
                    if np.count_nonzero(accepted)>=7:
                        displacement = moved[accepted]-old[accepted]
                        dx,dy = np.median(displacement,axis=0)
                        if abs(dx)<40 and abs(dy)<40:
                            self.last_boxes = [
                                (x1+dx,y1+dy,x2+dx,y2+dy)
                                for x1,y1,x2,y2 in self.last_boxes
                            ]
                            self.points = tracked[accepted].reshape(-1,1,2)
                        else:self.points=None
                    else:self.points=None
                else:self.points=None
            except cv.error:
                self.points=None
        self.prev_gray=gray
        tracked_ok=(bool(self.last_boxes) and self.points is not None
                    and now-self.last_detect_ok<COAST_LIMIT_S)
        if tracked_ok:
            # Light background mosaic plus stronger anonymous regions.
            result=_whole_mosaic(cv,image,2)
            for box in self.last_boxes:
                _mosaic_face(cv,result,box,self.cfg.pixelate_block)
        else:
            self.fallback_frames+=1
            result=_whole_mosaic(cv,image,self.cfg.fallback_block)
        ok,encoded=cv.imencode(
            ".jpg",result,[cv.IMWRITE_JPEG_QUALITY,self.cfg.jpeg_quality],
        )
        if not ok or encoded is None:
            raise MosaicDemoError("jpeg_encode_failed")
        # Never return jpeg or raw pixels from the input path.
        return encoded.tobytes(),len(self.last_boxes) if tracked_ok else 0,not tracked_ok


__all__ = [
    "DemoConfig","MosaicDemoError","SmoothMosaic","validate_model",
]
