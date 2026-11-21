"""Fail-closed JPEG validation and face redaction, without ROS or model sessions."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import asdict, dataclass

import numpy as np
from utils.cv2_compat import load_cv2

MAX_JPEG_BYTES = 8 * 1024 * 1024
MAX_IMAGE_SIDE = 8192
MAX_IMAGE_PIXELS = 16 * 1024 * 1024
MAX_DETECTIONS = 16800  # all anchors of the official 640x640 SCRFD


class PrivacyError(ValueError):
    """The frame cannot be released on the privacy output."""


@dataclass(frozen=True)
class PrivacyConfig:
    mode: str = "opaque"
    detect_fps: float = 10.0
    confidence_threshold: float = 0.5
    bbox_padding: float = 0.25
    pixelate_strength: int = 16
    gaussian_strength: int = 51
    inference_timeout_s: float = 2.0
    max_frame_age_s: float = 2.0

    def updated(self, values: dict) -> PrivacyConfig:
        if not isinstance(values, dict) or set(values) - set(asdict(self)):
            raise PrivacyError("unknown privacy configuration field")
        cfg = PrivacyConfig(**{**asdict(self), **values})
        if cfg.mode not in ("opaque", "pixelate", "gaussian"):
            raise PrivacyError("mode must be opaque, pixelate or gaussian")
        for name, low, high in (
            ("detect_fps", 0.1, 60),
            ("confidence_threshold", 0.1, 0.9),
            ("bbox_padding", 0, 1),
            ("inference_timeout_s", 0.05, 10),
            ("max_frame_age_s", 0.1, 10),
        ):
            value = getattr(cfg, name)
            if (
                type(value) not in (int, float)
                or not low <= value <= high
                or not math.isfinite(value)
            ):
                raise PrivacyError(f"{name} must be finite in [{low}, {high}]")
        for name, low, high in (
            ("pixelate_strength", 4, 128),
            ("gaussian_strength", 3, 301),
        ):
            value = getattr(cfg, name)
            if type(value) is not int or not low <= value <= high:
                raise PrivacyError(f"{name} must be an integer in [{low}, {high}]")
        if cfg.gaussian_strength % 2 != 1:
            raise PrivacyError("gaussian_strength must be odd")
        return cfg


def decode_jpeg(data: bytes) -> np.ndarray:
    """Inspect dimensions before allocating pixels; never auto-rotate EXIF.

    The parent and worker use this same decoder, so boxes refer to exactly
    the pixels that are redacted. No photo-format or raw-JPEG fallback.
    """
    if (
        not isinstance(data, bytes)
        or not 4 <= len(data) <= MAX_JPEG_BYTES
        or not data.startswith(b"\xff\xd8")
        or not data.endswith(b"\xff\xd9")
    ):
        raise PrivacyError("invalid_jpeg")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if (
                image.format != "JPEG"
                or min(width, height) < 1
                or max(width, height) > MAX_IMAGE_SIDE
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise PrivacyError("invalid_image_dimensions")
            image.verify()
        cv2 = load_cv2()
        decoded = cv2.imdecode(
            np.frombuffer(data, np.uint8),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        if decoded is None or decoded.shape != (height, width, 3):
            raise PrivacyError("jpeg_decode_failed")
        return decoded
    except PrivacyError:
        raise
    except Exception as exc:
        raise PrivacyError("jpeg_decode_failed") from exc


def validated_regions(
    faces, width: int, height: int, threshold: float, padding: float
) -> list[tuple[int, int, int, int]]:
    """Reject the whole result if any detection is malformed; cover all boxes.

    Coordinates may extend outside the image, but must have a nonempty
    intersection. Floor/ceil and padding cover subpixel and edge detections.
    """
    if not isinstance(faces, list) or len(faces) > MAX_DETECTIONS:
        raise PrivacyError("invalid_detection_list")
    regions = []
    for face in faces:
        if not isinstance(face, dict):
            raise PrivacyError("invalid_detection")
        box, score = face.get("bbox"), face.get("det_score")
        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or any(
                type(v) not in (int, float) or abs(v) > 10**9 or not math.isfinite(v)
                for v in box
            )
            or type(score) not in (int, float)
            or not threshold <= score <= 1
            or not math.isfinite(score)
        ):
            raise PrivacyError("invalid_detection")
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise PrivacyError("invalid_bbox")
        # Reject off-image boxes before padding can disguise them as valid.
        if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
            raise PrivacyError("off_image_bbox")
        dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
        if not all(
            math.isfinite(v) for v in (dx, dy, x1 - dx, y1 - dy, x2 + dx, y2 + dy)
        ):
            raise PrivacyError("invalid_bbox")
        regions.append(
            (
                max(0, math.floor(x1 - dx)),
                max(0, math.floor(y1 - dy)),
                min(width, math.ceil(x2 + dx)),
                min(height, math.ceil(y2 + dy)),
            )
        )
    return regions


def validate_reply(reply, data: bytes, token: str, shape, cfg: PrivacyConfig):
    if (
        not isinstance(reply, dict)
        or reply.get("ok") is not True
        or reply.get("complete") is not True
        or reply.get("truncated") is not False
    ):
        raise PrivacyError("incomplete_detection")
    height, width = shape[:2]
    if (
        reply.get("frame_token") != token
        or reply.get("sha256") != hashlib.sha256(data).hexdigest()
        or type(reply.get("width")) is not int
        or type(reply.get("height")) is not int
        or (reply["height"], reply["width"]) != (height, width)
    ):
        raise PrivacyError("frame_mismatch")
    return validated_regions(
        reply.get("faces"), width, height, cfg.confidence_threshold, cfg.bbox_padding
    )


def redact(image: np.ndarray, regions, cfg: PrivacyConfig) -> np.ndarray:
    """Always copy. Read each overlapping ROI from the source, write its union.

    Tiny regions in the weaker modes are blacked out. Opaque is the default;
    pixelation and Gaussian blur do not provide equivalent privacy strength.
    """
    cv2 = load_cv2()
    output = image.copy()
    for x1, y1, x2, y2 in regions:
        roi = image[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        if cfg.mode == "opaque" or min(w, h) < 3:
            output[y1:y2, x1:x2] = 0
        elif cfg.mode == "pixelate":
            small = cv2.resize(
                roi,
                (
                    max(1, w // cfg.pixelate_strength),
                    max(1, h // cfg.pixelate_strength),
                ),
                interpolation=cv2.INTER_AREA,
            )
            output[y1:y2, x1:x2] = cv2.resize(
                small, (w, h), interpolation=cv2.INTER_NEAREST
            )
        elif cfg.mode == "gaussian":
            kernel = cfg.gaussian_strength
            output[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel, kernel), 0)
        else:
            raise PrivacyError("invalid_mode")
    return output


def encode_jpeg(image: np.ndarray) -> bytes:
    cv2 = load_cv2()
    try:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok or encoded is None:
            raise PrivacyError("jpeg_encode_failed")
        data = encoded.tobytes()
        # Validate the actual encoder output before exposing it to consumers.
        if decode_jpeg(data).shape != image.shape:
            raise PrivacyError("jpeg_encode_failed")
        return data
    except Exception as exc:
        raise PrivacyError("jpeg_encode_failed") from exc
