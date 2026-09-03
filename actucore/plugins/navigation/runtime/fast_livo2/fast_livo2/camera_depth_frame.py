"""Decode the Driver-owned PSE1 depth envelope used by data collection."""

from __future__ import annotations

import json
import math
import struct
import zlib


MAGIC = b"PSE1"
SCHEMA = "phanthy.sensor.camera_depth_frame.v1"
ENVELOPE_FORMAT = "application/vnd.phanthy.sensor-envelope.v1"
_HEADER = struct.Struct("<4sII")
_MAX_METADATA_BYTES = 64 * 1024
_MAX_DEPTH_BYTES = 32 * 1024 * 1024
_DEPTH_COMPRESSION_CODEC = "zlib"
_DEPTH_COMPRESSION_LEVEL = 1
_DEPTH_UNIT = "realsense_depth_unit"
_DEPTH_SCALE_SEMANTICS = "meters_per_realsense_depth_unit"


class InvalidCameraDepthFrame(ValueError):
    pass


def _object(value: object, *, field: str) -> dict:
    if not isinstance(value, dict):
        raise InvalidCameraDepthFrame(f"{field} must be an object")
    return dict(value)


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise InvalidCameraDepthFrame(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise InvalidCameraDepthFrame(f"{field} must be a positive integer") from exc
    if result <= 0:
        raise InvalidCameraDepthFrame(f"{field} must be a positive integer")
    return result


def _positive_float(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidCameraDepthFrame(f"{field} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise InvalidCameraDepthFrame(f"{field} must be positive and finite")
    return result


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidCameraDepthFrame(f"{field} must be a non-empty string")
    return value.strip()


def _matrix(value: object, *, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 16:
        raise InvalidCameraDepthFrame(f"{field} must contain 16 finite numbers")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise InvalidCameraDepthFrame(
            f"{field} must contain 16 finite numbers"
        ) from exc
    if not all(math.isfinite(item) for item in result):
        raise InvalidCameraDepthFrame(f"{field} must contain 16 finite numbers")
    if any(
        abs(actual - expected) > 1e-6
        for actual, expected in zip(result[12:], (0.0, 0.0, 0.0, 1.0))
    ):
        raise InvalidCameraDepthFrame(f"{field} homogeneous bottom row is invalid")
    return result


def _transform(value: object, *, field: str) -> list[float]:
    entry = _object(value, field=field)
    if entry.get("status") == "unavailable":
        raise InvalidCameraDepthFrame(f"{field} is unavailable")
    transform = _object(entry.get("transform", entry), field=f"{field}.transform")
    if transform.get("convention") != "target_from_source":
        raise InvalidCameraDepthFrame(
            f"{field}.transform convention must be target_from_source"
        )
    return _matrix(
        transform.get("matrix_row_major"),
        field=f"{field}.transform.matrix_row_major",
    )


def validate_metadata(value: object) -> dict:
    metadata = _object(value, field="metadata")
    if metadata.get("schema") != SCHEMA:
        raise InvalidCameraDepthFrame(f"schema must be {SCHEMA}")
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
        raise InvalidCameraDepthFrame(
            "header.stamp_ns must equal timing.source_stamp_ns"
        )
    if timing.get("clock_domain") != "ros_system_time":
        raise InvalidCameraDepthFrame("timing.clock_domain must be ros_system_time")
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
    uncompressed_size = _positive_int(
        image.get("uncompressed_size"), field="image.uncompressed_size"
    )
    compression = _object(image.get("compression"), field="image.compression")
    if image.get("encoding") != "z16_le":
        raise InvalidCameraDepthFrame("image.encoding must be z16_le")
    expected_uncompressed_size = step_bytes * height
    if (
        step_bytes != width * 2
        or uncompressed_size != expected_uncompressed_size
        or uncompressed_size > _MAX_DEPTH_BYTES
        or payload_size > _MAX_DEPTH_BYTES
    ):
        raise InvalidCameraDepthFrame("depth dimensions, step, and payload size disagree")
    if compression.get("codec") != _DEPTH_COMPRESSION_CODEC:
        raise InvalidCameraDepthFrame(
            f"image.compression.codec must be {_DEPTH_COMPRESSION_CODEC}"
        )
    if compression.get("level") != _DEPTH_COMPRESSION_LEVEL:
        raise InvalidCameraDepthFrame(
            f"image.compression.level must be {_DEPTH_COMPRESSION_LEVEL}"
        )
    depth_scale_m = _positive_float(
        image.get("depth_scale_m"), field="image.depth_scale_m"
    )
    if image.get("unit") != _DEPTH_UNIT:
        raise InvalidCameraDepthFrame(f"image.unit must be {_DEPTH_UNIT}")
    if image.get("depth_scale_semantics") != _DEPTH_SCALE_SEMANTICS:
        raise InvalidCameraDepthFrame(
            f"image.depth_scale_semantics must be {_DEPTH_SCALE_SEMANTICS}"
        )
    if image.get("aligned_to_rgb") is not False:
        raise InvalidCameraDepthFrame("image.aligned_to_rgb must be false")

    calibration_id = _text(
        calibration.get("calibration_id"), field="calibration.calibration_id"
    )
    if (
        _positive_int(calibration.get("width"), field="calibration.width"),
        _positive_int(calibration.get("height"), field="calibration.height"),
    ) != (width, height):
        raise InvalidCameraDepthFrame(
            "calibration dimensions must match image dimensions"
        )
    calibration_scale = _positive_float(
        calibration.get("depth_scale_m"), field="calibration.depth_scale_m"
    )
    if not math.isclose(depth_scale_m, calibration_scale, rel_tol=0.0, abs_tol=1e-12):
        raise InvalidCameraDepthFrame("image and calibration depth scales disagree")
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
        "payload_size": payload_size,
        "uncompressed_size": uncompressed_size,
        "compression": {
            "codec": _DEPTH_COMPRESSION_CODEC,
            "level": _DEPTH_COMPRESSION_LEVEL,
        },
        "unit": _DEPTH_UNIT,
        "depth_scale_semantics": _DEPTH_SCALE_SEMANTICS,
        "depth_scale_m": depth_scale_m,
        "aligned_to_rgb": False,
        "envelope_format": ENVELOPE_FORMAT,
    }


def decode(payload: bytes) -> tuple[dict, bytes]:
    raw = bytes(payload)
    if len(raw) < _HEADER.size:
        raise InvalidCameraDepthFrame("camera depth frame payload is truncated")
    magic, metadata_size, payload_size = _HEADER.unpack_from(raw)
    if magic != MAGIC:
        raise InvalidCameraDepthFrame("camera depth frame magic is invalid")
    if metadata_size <= 0 or metadata_size > _MAX_METADATA_BYTES:
        raise InvalidCameraDepthFrame("camera depth frame metadata size is invalid")
    if payload_size <= 0 or payload_size > _MAX_DEPTH_BYTES:
        raise InvalidCameraDepthFrame("camera depth frame payload size is invalid")
    expected_size = _HEADER.size + metadata_size + payload_size
    if len(raw) != expected_size:
        raise InvalidCameraDepthFrame(
            f"camera depth frame envelope length mismatch: expected {expected_size}, got {len(raw)}"
        )
    metadata_end = _HEADER.size + metadata_size
    try:
        driver_metadata = json.loads(
            raw[_HEADER.size:metadata_end].decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCameraDepthFrame(
            "camera depth frame metadata is invalid JSON"
        ) from exc
    metadata = validate_metadata(driver_metadata)
    compressed_depth = raw[metadata_end:]
    if len(compressed_depth) != metadata["payload_size"]:
        raise InvalidCameraDepthFrame(
            "depth compressed payload length does not match metadata"
        )
    expected_size = metadata["uncompressed_size"]
    try:
        decompressor = zlib.decompressobj()
        depth = decompressor.decompress(compressed_depth, expected_size + 1)
        if decompressor.unconsumed_tail or len(depth) > expected_size:
            raise InvalidCameraDepthFrame(
                "depth decompressed payload exceeds declared size"
            )
        depth += decompressor.flush(expected_size - len(depth) + 1)
    except zlib.error as exc:
        raise InvalidCameraDepthFrame("depth zlib payload is invalid") from exc
    if (
        not decompressor.eof
        or decompressor.unused_data
        or len(depth) != expected_size
    ):
        raise InvalidCameraDepthFrame(
            "depth decompressed payload length does not match metadata"
        )
    return metadata, depth


def encode(metadata: dict, depth: bytes) -> bytes:
    """Encode a Driver-shaped fixture; production encoding remains Driver-owned."""

    raw_depth = bytes(depth)
    driver_metadata = dict(metadata)
    image = _object(driver_metadata.get("image"), field="image")
    compressed_depth = zlib.compress(raw_depth, level=_DEPTH_COMPRESSION_LEVEL)
    image["uncompressed_size"] = len(raw_depth)
    image["payload_size"] = len(compressed_depth)
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
        raise InvalidCameraDepthFrame("camera depth frame metadata is too large")
    return (
        _HEADER.pack(MAGIC, len(encoded), len(compressed_depth))
        + encoded
        + compressed_depth
    )


__all__ = [
    "ENVELOPE_FORMAT",
    "InvalidCameraDepthFrame",
    "MAGIC",
    "SCHEMA",
    "decode",
    "encode",
    "validate_metadata",
]
