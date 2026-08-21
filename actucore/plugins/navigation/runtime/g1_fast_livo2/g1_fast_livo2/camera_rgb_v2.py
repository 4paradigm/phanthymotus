"""Decode the released Driver camera RGB v2 envelope for offline annotation.

The ROS carrier is ``std_msgs/msg/UInt8MultiArray`` on the Driver-owned topic
``/ubuntu/navigation/camera/rgb``. Its payload is the existing sensor envelope:

``PSE2`` + little-endian uint32 JSON length + uint32 payload length + JSON + JPEG.

Driver metadata stays authoritative. This adapter validates its nested
header/timing/image/calibration objects and returns the flat internal view used
by the recorder and annotation pipeline.
"""

from __future__ import annotations

import json
import math
import struct


MAGIC = b"PSE2"
SCHEMA = "phanthy.sensor.camera_rgb.v2"
ENVELOPE_FORMAT = "application/vnd.phanthy.sensor-envelope.v2"
_HEADER = struct.Struct("<4sII")
_MAX_METADATA_BYTES = 64 * 1024
_MAX_JPEG_BYTES = 32 * 1024 * 1024

# Keep this aligned with adapter_node.py: the odometry consumed by annotation
# describes base_link, while Driver calibration describes livox_frame->camera.
_BASE_TO_LIDAR_XYZ_RPY = (
    -0.00368,
    0.00003,
    0.46018,
    0.0,
    0.04014257279586953,
    0.0,
)


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


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise InvalidCameraRgbV2(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise InvalidCameraRgbV2(f"{field} must be a positive integer") from exc
    if result <= 0:
        raise InvalidCameraRgbV2(f"{field} must be a positive integer")
    return result


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidCameraRgbV2(f"{field} must be a non-empty string")
    return value.strip()


def _object(value: object, *, field: str) -> dict:
    if not isinstance(value, dict):
        raise InvalidCameraRgbV2(f"{field} must be an object")
    return dict(value)


def _matrix(value: object, *, field: str) -> list[float]:
    matrix = _finite_numbers(value, count=16, field=field)
    if any(
        abs(a - b) > 1e-6
        for a, b in zip(matrix[12:], (0.0, 0.0, 0.0, 1.0))
    ):
        raise InvalidCameraRgbV2(f"{field} homogeneous bottom row is invalid")
    return matrix


def _matmul(left: list[float], right: list[float]) -> list[float]:
    return [
        sum(left[row * 4 + k] * right[k * 4 + column] for k in range(4))
        for row in range(4)
        for column in range(4)
    ]


def _rigid_inverse(matrix: list[float]) -> list[float]:
    rotation = [
        matrix[0], matrix[4], matrix[8],
        matrix[1], matrix[5], matrix[9],
        matrix[2], matrix[6], matrix[10],
    ]
    translation = (matrix[3], matrix[7], matrix[11])
    inverse_translation = [
        -sum(
            rotation[row * 3 + column] * translation[column]
            for column in range(3)
        )
        for row in range(3)
    ]
    return [
        rotation[0], rotation[1], rotation[2], inverse_translation[0],
        rotation[3], rotation[4], rotation[5], inverse_translation[1],
        rotation[6], rotation[7], rotation[8], inverse_translation[2],
        0.0, 0.0, 0.0, 1.0,
    ]


def _base_to_lidar_matrix() -> list[float]:
    x, y, z, roll, pitch, yaw = _BASE_TO_LIDAR_XYZ_RPY
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        x,
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        y,
        -sp,
        cp * sr,
        cp * cr,
        z,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _transform_matrix(value: object, *, field: str) -> list[float]:
    entry = _object(value, field=field)
    if entry.get("status") == "unavailable":
        raise InvalidCameraRgbV2(f"{field} is unavailable")
    transform = _object(
        entry.get("transform", entry), field=f"{field}.transform"
    )
    if transform.get("convention") != "target_from_source":
        raise InvalidCameraRgbV2(
            f"{field}.transform convention must be target_from_source"
        )
    return _matrix(
        transform.get("matrix_row_major"),
        field=f"{field}.transform.matrix_row_major",
    )


def validate_metadata(value: object) -> dict:
    metadata = _object(value, field="metadata")
    if metadata.get("schema") != SCHEMA:
        raise InvalidCameraRgbV2(f"schema must be {SCHEMA}")

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
        raise InvalidCameraRgbV2(
            "header.stamp_ns must equal timing.source_stamp_ns"
        )
    if timing.get("clock_domain") != "ros_system_time":
        raise InvalidCameraRgbV2("timing.clock_domain must be ros_system_time")
    receive_stamp_ns = _positive_int(
        timing.get("driver_receive_stamp_ns"),
        field="timing.driver_receive_stamp_ns",
    )
    frame_id = _text(header.get("frame_id"), field="header.frame_id")

    width = _positive_int(image.get("width"), field="image.width")
    height = _positive_int(image.get("height"), field="image.height")
    if width > 16384 or height > 16384:
        raise InvalidCameraRgbV2("image dimensions must be within [1, 16384]")
    if image.get("encoding") != "jpeg":
        raise InvalidCameraRgbV2("image.encoding must be jpeg")

    calibration_id = _text(
        calibration.get("calibration_id"), field="calibration.calibration_id"
    )
    calibration_width = _positive_int(
        calibration.get("width"), field="calibration.width"
    )
    calibration_height = _positive_int(
        calibration.get("height"), field="calibration.height"
    )
    if (calibration_width, calibration_height) != (width, height):
        raise InvalidCameraRgbV2(
            "calibration dimensions must match image dimensions"
        )
    k = _finite_numbers(
        calibration.get("k"), count=9, field="calibration.k"
    )
    fx, fy, cx, cy = k[0], k[4], k[2], k[5]
    if fx <= 0.0 or fy <= 0.0:
        raise InvalidCameraRgbV2("calibration focal lengths must be positive")
    distortion_model = str(
        calibration.get("distortion_model", "none")
    ).strip().lower()
    supported_models = {
        "none",
        "plumb_bob",
        "brown_conrady",
        "inverse_brown_conrady",
        "realsense_inverse_brown_conrady",
        "rational_polynomial",
    }
    if distortion_model not in supported_models:
        raise InvalidCameraRgbV2(
            "calibration.distortion_model must be one of "
            + ",".join(sorted(supported_models))
        )
    coefficients = calibration.get("d", [])
    if not isinstance(coefficients, list) or len(coefficients) > 16:
        raise InvalidCameraRgbV2(
            "calibration.d must contain at most 16 numbers"
        )
    coefficients = _finite_numbers(
        coefficients, count=len(coefficients), field="calibration.d"
    )
    minimum_coefficients = {
        "none": 0,
        "plumb_bob": 5,
        "brown_conrady": 5,
        "inverse_brown_conrady": 5,
        "realsense_inverse_brown_conrady": 5,
        "rational_polynomial": 8,
    }[distortion_model]
    if len(coefficients) < minimum_coefficients:
        raise InvalidCameraRgbV2(
            "calibration.d is too short for " + distortion_model
        )

    t_camera_lidar = _transform_matrix(
        calibration.get("lidar_to_camera"),
        field="calibration.lidar_to_camera",
    )
    base_to_camera = calibration.get("base_to_camera")
    if base_to_camera is None:
        t_base_camera = _matmul(
            _base_to_lidar_matrix(),
            _rigid_inverse(t_camera_lidar),
        )
        base_transform_source = (
            "actucore_g1_base_to_lidar+driver_lidar_to_camera"
        )
    else:
        t_base_camera = _transform_matrix(
            base_to_camera,
            field="calibration.base_to_camera",
        )
        base_transform_source = "driver_calibration"

    sequence = metadata.get("sequence")
    if isinstance(sequence, bool):
        raise InvalidCameraRgbV2("sequence must be a non-negative integer")
    try:
        sequence = int(sequence)
    except (TypeError, ValueError) as exc:
        raise InvalidCameraRgbV2(
            "sequence must be a non-negative integer"
        ) from exc
    if sequence < 0:
        raise InvalidCameraRgbV2("sequence must be a non-negative integer")

    return {
        "schema": SCHEMA,
        "source_stamp_ns": source_stamp_ns,
        "receive_stamp_ns": receive_stamp_ns,
        "frame_id": frame_id,
        "calibration_id": calibration_id,
        "width": width,
        "height": height,
        "encoding": "jpeg",
        "sequence": sequence,
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "distortion_model": distortion_model,
            "coefficients": coefficients,
        },
        "t_camera_lidar": t_camera_lidar,
        "t_base_camera": t_base_camera,
        "base_transform_source": base_transform_source,
        "envelope_format": ENVELOPE_FORMAT,
    }


def _validate_jpeg(image: bytes) -> bytes:
    if (
        len(image) < 4
        or len(image) > _MAX_JPEG_BYTES
        or not image.startswith(b"\xff\xd8")
        or not image.endswith(b"\xff\xd9")
    ):
        raise InvalidCameraRgbV2("camera RGB v2 JPEG size is invalid")
    return image


def decode(payload: bytes) -> tuple[dict, bytes]:
    raw = bytes(payload)
    if len(raw) < _HEADER.size:
        raise InvalidCameraRgbV2("camera RGB v2 payload is truncated")
    magic, metadata_size, payload_size = _HEADER.unpack_from(raw)
    if magic != MAGIC:
        raise InvalidCameraRgbV2("camera RGB v2 magic is invalid")
    if metadata_size <= 0 or metadata_size > _MAX_METADATA_BYTES:
        raise InvalidCameraRgbV2("camera RGB v2 metadata size is invalid")
    if payload_size <= 0 or payload_size > _MAX_JPEG_BYTES:
        raise InvalidCameraRgbV2("camera RGB v2 payload size is invalid")
    expected_size = _HEADER.size + metadata_size + payload_size
    if len(raw) != expected_size:
        raise InvalidCameraRgbV2(
            "camera RGB v2 envelope length mismatch: "
            f"expected {expected_size}, got {len(raw)}"
        )
    metadata_end = _HEADER.size + metadata_size
    try:
        driver_metadata = json.loads(
            raw[_HEADER.size:metadata_end].decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCameraRgbV2(
            "camera RGB v2 metadata is invalid JSON"
        ) from exc
    metadata = validate_metadata(driver_metadata)
    jpeg = _validate_jpeg(raw[metadata_end:])
    image = _object(driver_metadata.get("image"), field="image")
    declared_payload_size = _positive_int(
        image.get("payload_size"), field="image.payload_size"
    )
    if declared_payload_size != payload_size:
        raise InvalidCameraRgbV2(
            "image.payload_size must equal the envelope payload size"
        )
    return metadata, jpeg


def encode(metadata: dict, jpeg: bytes) -> bytes:
    """Encode a Driver-shaped fixture; production encoding remains Driver-owned."""

    image = _validate_jpeg(bytes(jpeg))
    validate_metadata(metadata)
    driver_metadata = dict(metadata)
    image_metadata = _object(driver_metadata.get("image"), field="image")
    image_metadata["payload_size"] = len(image)
    driver_metadata["image"] = image_metadata
    encoded = json.dumps(
        driver_metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _MAX_METADATA_BYTES:
        raise InvalidCameraRgbV2("camera RGB v2 metadata is too large")
    return _HEADER.pack(MAGIC, len(encoded), len(image)) + encoded + image


__all__ = [
    "ENVELOPE_FORMAT",
    "InvalidCameraRgbV2",
    "MAGIC",
    "SCHEMA",
    "decode",
    "encode",
    "validate_metadata",
]
