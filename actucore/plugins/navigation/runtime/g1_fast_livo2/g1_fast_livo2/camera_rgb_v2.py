"""Versioned RGB envelope shared by Driver capture and offline annotation.

The ROS carrier is ``std_msgs/msg/UInt8MultiArray``.  Its payload is:

``CRGBV2\0\0`` + little-endian uint32 JSON length + UTF-8 JSON + JPEG bytes.

The JSON metadata is deliberately self-contained so a recorded image keeps the
calibration and timestamps needed for deterministic offline projection.
"""

from __future__ import annotations

import json
import math
import struct


MAGIC = b"CRGBV2\0\0"
SCHEMA = "phanthy.sensor.camera_rgb.v2"
_HEADER = struct.Struct("<8sI")
_MAX_METADATA_BYTES = 64 * 1024
_MAX_JPEG_BYTES = 32 * 1024 * 1024


class InvalidCameraRgbV2(ValueError):
    pass


def _finite_numbers(values, *, count: int, field: str) -> list[float]:
    if not isinstance(values, list) or len(values) != count:
        raise InvalidCameraRgbV2(f"{field} must contain {count} numbers")
    result = []
    for value in values:
        if isinstance(value, bool):
            raise InvalidCameraRgbV2(f"{field} must contain finite numbers")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidCameraRgbV2(
                f"{field} must contain finite numbers"
            ) from exc
        if not math.isfinite(number):
            raise InvalidCameraRgbV2(f"{field} must contain finite numbers")
        result.append(number)
    return result


def validate_metadata(value: object) -> dict:
    if not isinstance(value, dict):
        raise InvalidCameraRgbV2("metadata must be an object")
    metadata = dict(value)
    if metadata.get("schema") != SCHEMA:
        raise InvalidCameraRgbV2(f"schema must be {SCHEMA}")
    for field in ("source_stamp_ns", "receive_stamp_ns"):
        raw = metadata.get(field)
        if isinstance(raw, bool):
            raise InvalidCameraRgbV2(f"{field} must be a positive integer")
        try:
            stamp = int(raw)
        except (TypeError, ValueError) as exc:
            raise InvalidCameraRgbV2(
                f"{field} must be a positive integer"
            ) from exc
        if stamp <= 0:
            raise InvalidCameraRgbV2(f"{field} must be a positive integer")
        metadata[field] = stamp
    for field in ("width", "height"):
        raw = metadata.get(field)
        if isinstance(raw, bool):
            raise InvalidCameraRgbV2(f"{field} must be a positive integer")
        try:
            dimension = int(raw)
        except (TypeError, ValueError) as exc:
            raise InvalidCameraRgbV2(
                f"{field} must be a positive integer"
            ) from exc
        if not 1 <= dimension <= 16384:
            raise InvalidCameraRgbV2(f"{field} must be within [1, 16384]")
        metadata[field] = dimension
    for field in ("frame_id", "calibration_id"):
        text = metadata.get(field)
        if not isinstance(text, str) or not text.strip():
            raise InvalidCameraRgbV2(f"{field} must be a non-empty string")
        metadata[field] = text.strip()
    intrinsics = metadata.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise InvalidCameraRgbV2("intrinsics must be an object")
    normalized_intrinsics = {
        key: _finite_numbers([intrinsics.get(key)], count=1, field=f"intrinsics.{key}")[0]
        for key in ("fx", "fy", "cx", "cy")
    }
    if normalized_intrinsics["fx"] <= 0.0 or normalized_intrinsics["fy"] <= 0.0:
        raise InvalidCameraRgbV2("intrinsics fx/fy must be positive")
    distortion_model = str(
        intrinsics.get("distortion_model", "none")
    ).strip().lower()
    supported_models = {"none", "plumb_bob", "brown_conrady", "rational_polynomial"}
    if distortion_model not in supported_models:
        raise InvalidCameraRgbV2(
            "intrinsics.distortion_model must be one of "
            + ",".join(sorted(supported_models))
        )
    normalized_intrinsics["distortion_model"] = distortion_model
    coefficients = intrinsics.get("coefficients", [])
    if not isinstance(coefficients, list) or len(coefficients) > 16:
        raise InvalidCameraRgbV2(
            "intrinsics.coefficients must contain at most 16 numbers"
        )
    normalized_intrinsics["coefficients"] = _finite_numbers(
        coefficients,
        count=len(coefficients),
        field="intrinsics.coefficients",
    )
    minimum_coefficients = {
        "none": 0,
        "plumb_bob": 5,
        "brown_conrady": 5,
        "rational_polynomial": 8,
    }[distortion_model]
    if len(coefficients) < minimum_coefficients:
        raise InvalidCameraRgbV2(
            "intrinsics.coefficients is too short for " + distortion_model
        )
    metadata["intrinsics"] = normalized_intrinsics
    metadata["t_camera_lidar"] = _finite_numbers(
        metadata.get("t_camera_lidar"), count=16, field="t_camera_lidar"
    )
    metadata["t_base_camera"] = _finite_numbers(
        metadata.get("t_base_camera"), count=16, field="t_base_camera"
    )
    if metadata.get("encoding") != "jpeg":
        raise InvalidCameraRgbV2("encoding must be jpeg")
    return metadata


def decode(payload: bytes) -> tuple[dict, bytes]:
    raw = bytes(payload)
    if len(raw) < _HEADER.size:
        raise InvalidCameraRgbV2("camera RGB v2 payload is truncated")
    magic, metadata_size = _HEADER.unpack_from(raw)
    if magic != MAGIC:
        raise InvalidCameraRgbV2("camera RGB v2 magic is invalid")
    if metadata_size <= 0 or metadata_size > _MAX_METADATA_BYTES:
        raise InvalidCameraRgbV2("camera RGB v2 metadata size is invalid")
    metadata_end = _HEADER.size + metadata_size
    if metadata_end > len(raw):
        raise InvalidCameraRgbV2("camera RGB v2 metadata is truncated")
    jpeg = raw[metadata_end:]
    if (
        len(jpeg) < 4
        or len(jpeg) > _MAX_JPEG_BYTES
        or not jpeg.startswith(b"\xff\xd8")
        or not jpeg.endswith(b"\xff\xd9")
    ):
        raise InvalidCameraRgbV2("camera RGB v2 JPEG size is invalid")
    try:
        metadata = json.loads(raw[_HEADER.size:metadata_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCameraRgbV2("camera RGB v2 metadata is invalid JSON") from exc
    return validate_metadata(metadata), jpeg


def encode(metadata: dict, jpeg: bytes) -> bytes:
    normalized = validate_metadata(metadata)
    image = bytes(jpeg)
    if (
        len(image) < 4
        or len(image) > _MAX_JPEG_BYTES
        or not image.startswith(b"\xff\xd8")
        or not image.endswith(b"\xff\xd9")
    ):
        raise InvalidCameraRgbV2("camera RGB v2 JPEG size is invalid")
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_METADATA_BYTES:
        raise InvalidCameraRgbV2("camera RGB v2 metadata is too large")
    return _HEADER.pack(MAGIC, len(encoded)) + encoded + image


__all__ = [
    "InvalidCameraRgbV2",
    "MAGIC",
    "SCHEMA",
    "decode",
    "encode",
    "validate_metadata",
]
