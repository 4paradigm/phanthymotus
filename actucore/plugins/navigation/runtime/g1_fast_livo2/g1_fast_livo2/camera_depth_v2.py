"""Decode the Driver-owned PSE2 depth envelope used by data collection."""

from __future__ import annotations

import json
import math
import struct


MAGIC = b"PSE2"
SCHEMA = "phanthy.sensor.camera_depth.v2"
ENVELOPE_FORMAT = "application/vnd.phanthy.sensor-envelope.v2"
_HEADER = struct.Struct("<4sII")
_MAX_METADATA_BYTES = 64 * 1024
_MAX_DEPTH_BYTES = 32 * 1024 * 1024


class InvalidCameraDepthV2(ValueError):
    pass


def _object(value: object, *, field: str) -> dict:
    if not isinstance(value, dict):
        raise InvalidCameraDepthV2(f"{field} must be an object")
    return dict(value)


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise InvalidCameraDepthV2(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise InvalidCameraDepthV2(f"{field} must be a positive integer") from exc
    if result <= 0:
        raise InvalidCameraDepthV2(f"{field} must be a positive integer")
    return result


def _positive_float(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidCameraDepthV2(f"{field} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise InvalidCameraDepthV2(f"{field} must be positive and finite")
    return result


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidCameraDepthV2(f"{field} must be a non-empty string")
    return value.strip()


def _matrix(value: object, *, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 16:
        raise InvalidCameraDepthV2(f"{field} must contain 16 finite numbers")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise InvalidCameraDepthV2(
            f"{field} must contain 16 finite numbers"
        ) from exc
    if not all(math.isfinite(item) for item in result):
        raise InvalidCameraDepthV2(f"{field} must contain 16 finite numbers")
    if any(
        abs(actual - expected) > 1e-6
        for actual, expected in zip(result[12:], (0.0, 0.0, 0.0, 1.0))
    ):
        raise InvalidCameraDepthV2(f"{field} homogeneous bottom row is invalid")
    return result


def _transform(value: object, *, field: str) -> list[float]:
    entry = _object(value, field=field)
    if entry.get("status") == "unavailable":
        raise InvalidCameraDepthV2(f"{field} is unavailable")
    transform = _object(entry.get("transform", entry), field=f"{field}.transform")
    if transform.get("convention") != "target_from_source":
        raise InvalidCameraDepthV2(
            f"{field}.transform convention must be target_from_source"
        )
    return _matrix(
        transform.get("matrix_row_major"),
        field=f"{field}.transform.matrix_row_major",
    )


def validate_metadata(value: object) -> dict:
    metadata = _object(value, field="metadata")
    if metadata.get("schema") != SCHEMA:
        raise InvalidCameraDepthV2(f"schema must be {SCHEMA}")
    header = _object(metadata.get("header"), field="header")
    timing = _object(metadata.get("timing"), field="timing")
    image = _object(metadata.get("image"), field="image")
    calibration = _object(metadata.get("calibration"), field="calibration")

    source_stamp_ns = _positive_int(
        header.get("stamp_ns"), field="header.stamp_ns"
    )
    timing_stamp_ns = _positive_int(
        timing.get("source_stamp_ns"), field="timing.source_stamp_ns"
    )
    if source_stamp_ns != timing_stamp_ns:
        raise InvalidCameraDepthV2(
            "header.stamp_ns must equal timing.source_stamp_ns"
        )
    if timing.get("clock_domain") != "ros_system_time":
        raise InvalidCameraDepthV2("timing.clock_domain must be ros_system_time")
    receive_stamp_ns = _positive_int(
        timing.get("driver_receive_stamp_ns"),
        field="timing.driver_receive_stamp_ns",
    )
    width = _positive_int(image.get("width"), field="image.width")
    height = _positive_int(image.get("height"), field="image.height")
    step_bytes = _positive_int(image.get("step_bytes"), field="image.step_bytes")
    payload_size = _positive_int(
        image.get("payload_size"), field="image.payload_size"
    )
    if image.get("encoding") != "z16_le":
        raise InvalidCameraDepthV2("image.encoding must be z16_le")
    if step_bytes != width * 2 or payload_size != step_bytes * height:
        raise InvalidCameraDepthV2("depth dimensions, step, and payload size disagree")
    depth_scale_m = _positive_float(
        image.get("depth_scale_m"), field="image.depth_scale_m"
    )
    if image.get("unit") != "meter":
        raise InvalidCameraDepthV2("image.unit must be meter")
    if image.get("aligned_to_rgb") is not False:
        raise InvalidCameraDepthV2("image.aligned_to_rgb must be false")

    calibration_id = _text(
        calibration.get("calibration_id"), field="calibration.calibration_id"
    )
    if (
        _positive_int(calibration.get("width"), field="calibration.width"),
        _positive_int(calibration.get("height"), field="calibration.height"),
    ) != (width, height):
        raise InvalidCameraDepthV2(
            "calibration dimensions must match image dimensions"
        )
    calibration_scale = _positive_float(
        calibration.get("depth_scale_m"), field="calibration.depth_scale_m"
    )
    if not math.isclose(depth_scale_m, calibration_scale, rel_tol=0.0, abs_tol=1e-12):
        raise InvalidCameraDepthV2("image and calibration depth scales disagree")
    _transform(calibration.get("depth_to_rgb"), field="calibration.depth_to_rgb")
    _transform(
        calibration.get("lidar_to_camera"), field="calibration.lidar_to_camera"
    )
    _object(calibration.get("rgb_intrinsics"), field="calibration.rgb_intrinsics")

    return {
        "schema": SCHEMA,
        "source_stamp_ns": source_stamp_ns,
        "receive_stamp_ns": receive_stamp_ns,
        "frame_id": _text(header.get("frame_id"), field="header.frame_id"),
        "calibration_id": calibration_id,
        "width": width,
        "height": height,
        "encoding": "z16_le",
        "step_bytes": step_bytes,
        "depth_scale_m": depth_scale_m,
        "aligned_to_rgb": False,
        "envelope_format": ENVELOPE_FORMAT,
    }


def decode(payload: bytes) -> tuple[dict, bytes]:
    raw = bytes(payload)
    if len(raw) < _HEADER.size:
        raise InvalidCameraDepthV2("camera depth v2 payload is truncated")
    magic, metadata_size, payload_size = _HEADER.unpack_from(raw)
    if magic != MAGIC:
        raise InvalidCameraDepthV2("camera depth v2 magic is invalid")
    if metadata_size <= 0 or metadata_size > _MAX_METADATA_BYTES:
        raise InvalidCameraDepthV2("camera depth v2 metadata size is invalid")
    if payload_size <= 0 or payload_size > _MAX_DEPTH_BYTES:
        raise InvalidCameraDepthV2("camera depth v2 payload size is invalid")
    expected_size = _HEADER.size + metadata_size + payload_size
    if len(raw) != expected_size:
        raise InvalidCameraDepthV2(
            f"camera depth v2 envelope length mismatch: expected {expected_size}, got {len(raw)}"
        )
    metadata_end = _HEADER.size + metadata_size
    try:
        driver_metadata = json.loads(
            raw[_HEADER.size:metadata_end].decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCameraDepthV2(
            "camera depth v2 metadata is invalid JSON"
        ) from exc
    metadata = validate_metadata(driver_metadata)
    depth = raw[metadata_end:]
    if len(depth) != metadata["step_bytes"] * metadata["height"]:
        raise InvalidCameraDepthV2("depth payload length does not match metadata")
    return metadata, depth


def encode(metadata: dict, depth: bytes) -> bytes:
    """Encode a Driver-shaped fixture; production encoding remains Driver-owned."""

    raw_depth = bytes(depth)
    driver_metadata = dict(metadata)
    image = _object(driver_metadata.get("image"), field="image")
    image["payload_size"] = len(raw_depth)
    driver_metadata["image"] = image
    validate_metadata(driver_metadata)
    encoded = json.dumps(
        driver_metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _MAX_METADATA_BYTES:
        raise InvalidCameraDepthV2("camera depth v2 metadata is too large")
    return _HEADER.pack(MAGIC, len(encoded), len(raw_depth)) + encoded + raw_depth


__all__ = [
    "ENVELOPE_FORMAT",
    "InvalidCameraDepthV2",
    "MAGIC",
    "SCHEMA",
    "decode",
    "encode",
    "validate_metadata",
]
