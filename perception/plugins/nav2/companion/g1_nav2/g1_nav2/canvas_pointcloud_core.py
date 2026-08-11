"""Decode the timestamped G1 Driver point-cloud envelope for Nav2."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import struct
from typing import Any, Iterable

import numpy as np

from .timestamp_contract import (
    DEFAULT_MAX_FUTURE_SKEW_NS,
    DEFAULT_MAX_SOURCE_AGE_NS,
    InvalidSourceTimestamp,
    validate_source_timestamp_ns,
)


LIDAR_CLOUD_SCHEMA = "phanthy.g1.lidar_cloud.v2"
LIDAR_CLOUD_TRAILER_MAGIC = b"PCLMETA2"
DRIVER_POINT_DATA_TRANSFORM = "gravity_aligned_roll_pitch"
POINT_DATA_TRANSFORM = "rigid_lidar_frame_restored"
POINT_DATA_TRANSFORM_PARAMS_VERSION = 1
POINT_FIELD_FLOAT32 = 7
NSEC_PER_SEC = 1_000_000_000
_LEGACY_HEADER = struct.Struct("<II")
_TRAILER = struct.Struct("<I8s")
_MIN_POINT_STEP = 12
_MAX_POINT_STEP = 512
_MAX_POINTS = 2_000_000
_MAX_METADATA_BYTES = 1_048_576
_POINT_FIELD_SIZES = {
    1: 1,  # INT8
    2: 1,  # UINT8
    3: 2,  # INT16
    4: 2,  # UINT16
    5: 4,  # INT32
    6: 4,  # UINT32
    7: 4,  # FLOAT32
    8: 8,  # FLOAT64
}


class InvalidCanvasPointCloud(ValueError):
    """Raised when a G1 Driver point-cloud envelope is malformed or stale."""


@dataclass(frozen=True)
class CanvasPointField:
    name: str
    offset: int
    datatype: int
    count: int


@dataclass(frozen=True)
class CanvasPointCloud:
    source_stamp_ns: int
    driver_receive_monotonic_ns: int
    timestamp_source: str
    frame_id: str
    frame_source: str
    source_schema: str
    raw_lidar_stamp_ns: int | None
    raw_lidar_stamp_valid: bool
    raw_lidar_frame_id: str
    source_point_data_transform: str
    point_data_transform: str
    gravity_alignment_applied: bool
    gravity_alignment_roll_rad: float
    gravity_alignment_pitch_rad: float
    gravity_alignment_attitude_source: str
    gravity_alignment_attitude_time_correlated: bool
    height: int
    width: int
    fields: tuple[CanvasPointField, ...]
    is_bigendian: bool
    point_step: int
    row_step: int
    is_dense: bool
    point_count: int
    scan_end_offset_ns: int
    data: bytes


@dataclass(frozen=True)
class LidarClockSnapshot:
    ready: bool
    mode: str
    samples: int
    offset_ns: int | None
    residual_ns: int | None
    resets: int


class LidarClockNormalizer:
    """Map the native LiDAR scan clock into Driver system time.

    ``PCLMETA2`` gives both the native scan-start stamp and the Driver callback
    receive time.  The minimum observed ``receive - scan_end`` offset rejects
    callback scheduling jitter while retaining the physical scan-start time.
    A native timestamp regression resets the estimator and requires a fresh
    warm-up before another cloud can reach Nav2.
    """

    def __init__(
        self,
        *,
        mode: str = "auto",
        warmup_samples: int = 8,
        window_samples: int = 200,
        aligned_tolerance_ns: int = 2 * NSEC_PER_SEC,
    ) -> None:
        if mode not in {"auto", "normalize", "passthrough"}:
            raise ValueError("mode must be auto, normalize, or passthrough")
        if warmup_samples < 1:
            raise ValueError("warmup_samples must be positive")
        if window_samples < warmup_samples:
            raise ValueError("window_samples must be >= warmup_samples")
        if aligned_tolerance_ns < 1:
            raise ValueError("aligned_tolerance_ns must be positive")
        self._configured_mode = mode
        self._active_mode: str | None = None
        self._warmup_samples = int(warmup_samples)
        self._aligned_tolerance_ns = int(aligned_tolerance_ns)
        self._candidates: deque[int] = deque(maxlen=int(window_samples))
        self._last_raw_stamp_ns: int | None = None
        self._offset_ns: int | None = None
        self._residual_ns: int | None = None
        self._resets = 0

    def normalize(
        self,
        *,
        raw_stamp_ns: int,
        driver_receive_stamp_ns: int,
        scan_end_offset_ns: int,
    ) -> int | None:
        raw_stamp_ns = int(raw_stamp_ns)
        driver_receive_stamp_ns = int(driver_receive_stamp_ns)
        scan_end_offset_ns = int(scan_end_offset_ns)
        if raw_stamp_ns <= 0 or driver_receive_stamp_ns <= 0:
            raise ValueError("raw and Driver receive timestamps must be positive")
        if not 0 <= scan_end_offset_ns <= NSEC_PER_SEC:
            raise ValueError("scan_end_offset_ns must be within one second")

        if (
            self._last_raw_stamp_ns is not None
            and raw_stamp_ns <= self._last_raw_stamp_ns
        ):
            self.reset()
        self._last_raw_stamp_ns = raw_stamp_ns

        if self._active_mode is None:
            self._active_mode = self._resolve_mode(
                raw_stamp_ns=raw_stamp_ns,
                driver_receive_stamp_ns=driver_receive_stamp_ns,
            )

        if self._active_mode == "passthrough":
            raw_age_ns = driver_receive_stamp_ns - raw_stamp_ns
            if not (
                -self._aligned_tolerance_ns
                <= raw_age_ns
                <= self._aligned_tolerance_ns
            ):
                raise ValueError("passthrough LiDAR clock left system-time domain")
            self._offset_ns = 0
            self._residual_ns = raw_age_ns - scan_end_offset_ns
            return raw_stamp_ns

        candidate = (
            driver_receive_stamp_ns - raw_stamp_ns - scan_end_offset_ns
        )
        self._candidates.append(candidate)
        self._offset_ns = min(self._candidates)
        self._residual_ns = candidate - self._offset_ns
        if len(self._candidates) < self._warmup_samples:
            return None
        return raw_stamp_ns + self._offset_ns

    def snapshot(self) -> LidarClockSnapshot:
        return LidarClockSnapshot(
            ready=(
                self._active_mode == "passthrough"
                or len(self._candidates) >= self._warmup_samples
            ),
            mode=self._active_mode or "warming_up",
            samples=len(self._candidates),
            offset_ns=self._offset_ns,
            residual_ns=self._residual_ns,
            resets=self._resets,
        )

    def _resolve_mode(
        self, *, raw_stamp_ns: int, driver_receive_stamp_ns: int
    ) -> str:
        if self._configured_mode != "auto":
            return self._configured_mode
        raw_age_ns = driver_receive_stamp_ns - raw_stamp_ns
        if -self._aligned_tolerance_ns <= raw_age_ns <= self._aligned_tolerance_ns:
            return "passthrough"
        return "normalize"

    def reset(self) -> None:
        """Discard the clock estimate after a source/topology discontinuity."""

        self._candidates.clear()
        self._active_mode = None
        self._last_raw_stamp_ns = None
        self._offset_ns = None
        self._residual_ns = None
        self._resets += 1


def _valid_frame_id(frame_id: str) -> bool:
    return bool(
        frame_id
        and not frame_id.startswith("/")
        and "//" not in frame_id
        and frame_id.replace("/", "").replace("_", "").isalnum()
    )


def _require_int(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidCanvasPointCloud(f"{name} must be an integer")
    if positive and value <= 0:
        raise InvalidCanvasPointCloud(f"{name} must be positive")
    return value


def _validate_shape(point_step: int, point_count: int) -> None:
    if not _MIN_POINT_STEP <= point_step <= _MAX_POINT_STEP:
        raise InvalidCanvasPointCloud(f"invalid point_step: {point_step}")
    if not 0 < point_count <= _MAX_POINTS:
        raise InvalidCanvasPointCloud(f"invalid point_count: {point_count}")


def _decode_metadata(raw: bytes, *, point_data_end: int) -> dict[str, Any]:
    if len(raw) < point_data_end + _TRAILER.size:
        raise InvalidCanvasPointCloud("PCLMETA2 metadata footer is required")
    metadata_length, magic = _TRAILER.unpack_from(raw, len(raw) - _TRAILER.size)
    if magic != LIDAR_CLOUD_TRAILER_MAGIC:
        raise InvalidCanvasPointCloud("PCLMETA2 metadata footer is required")
    if not 0 < metadata_length <= _MAX_METADATA_BYTES:
        raise InvalidCanvasPointCloud(
            f"invalid PCLMETA2 metadata length: {metadata_length}"
        )
    metadata_end = len(raw) - _TRAILER.size
    metadata_start = metadata_end - metadata_length
    if metadata_start != point_data_end:
        raise InvalidCanvasPointCloud(
            "PCLMETA2 metadata does not immediately follow the legacy point data"
        )
    try:
        metadata = json.loads(raw[metadata_start:metadata_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCanvasPointCloud("PCLMETA2 metadata is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise InvalidCanvasPointCloud("PCLMETA2 metadata root must be an object")
    return metadata


def _decode_fields(value: Any, *, point_step: int) -> tuple[CanvasPointField, ...]:
    if not isinstance(value, list) or not value:
        raise InvalidCanvasPointCloud("pointcloud.fields must be a non-empty array")
    fields: list[CanvasPointField] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise InvalidCanvasPointCloud(f"pointcloud.fields[{index}] must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise InvalidCanvasPointCloud(f"invalid or duplicate point field: {name!r}")
        offset = _require_int(item.get("offset"), f"field {name}.offset")
        datatype = _require_int(item.get("datatype"), f"field {name}.datatype")
        count = _require_int(item.get("count"), f"field {name}.count", positive=True)
        field_size = _POINT_FIELD_SIZES.get(datatype)
        if field_size is None:
            raise InvalidCanvasPointCloud(
                f"field {name} has unsupported PointField datatype: {datatype}"
            )
        if offset < 0 or offset + field_size * count > point_step:
            raise InvalidCanvasPointCloud(f"field {name} is outside point_step")
        names.add(name)
        fields.append(CanvasPointField(name, offset, datatype, count))

    by_name = {field.name: field for field in fields}
    for name, offset in (("x", 0), ("y", 4), ("z", 8)):
        field = by_name.get(name)
        if (
            field is None
            or field.offset != offset
            or field.datatype != POINT_FIELD_FLOAT32
            or field.count != 1
        ):
            raise InvalidCanvasPointCloud(
                f"point field {name} must be FLOAT32 count=1 at offset {offset}"
            )
    time_field = by_name.get("time")
    if (
        time_field is None
        or time_field.datatype != POINT_FIELD_FLOAT32
        or time_field.count != 1
    ):
        raise InvalidCanvasPointCloud(
            "point field time must be FLOAT32 count=1"
        )
    return tuple(fields)


def point_time_max_offset_ns(
    *,
    data: bytes | bytearray | memoryview,
    fields: Iterable[CanvasPointField],
    height: int,
    width: int,
    point_step: int,
    row_step: int,
    is_bigendian: bool,
) -> int:
    """Return the MID360 scan-end offset from the per-point ``time`` field."""

    time_fields = [field for field in fields if field.name == "time"]
    if len(time_fields) != 1:
        raise InvalidCanvasPointCloud("PointCloud2 requires one time field")
    time_field = time_fields[0]
    if (
        time_field.datatype != POINT_FIELD_FLOAT32
        or time_field.count != 1
        or time_field.offset < 0
        or time_field.offset + 4 > point_step
    ):
        raise InvalidCanvasPointCloud("unexpected PointCloud2 time field layout")

    raw = memoryview(data)
    unpack = struct.Struct(">f" if is_bigendian else "<f").unpack_from
    maximum = 0.0
    try:
        for row in range(height):
            row_offset = row * row_step
            for column in range(width):
                value = float(
                    unpack(
                        raw,
                        row_offset + column * point_step + time_field.offset,
                    )[0]
                )
                if not math.isfinite(value) or value < 0.0:
                    raise InvalidCanvasPointCloud(
                        "PointCloud2 contains an invalid point time"
                    )
                maximum = max(maximum, value)
    except (struct.error, TypeError, ValueError) as exc:
        if isinstance(exc, InvalidCanvasPointCloud):
            raise
        raise InvalidCanvasPointCloud("invalid PointCloud2 time buffer") from exc
    maximum_ns = int(round(maximum))
    if maximum_ns > NSEC_PER_SEC:
        raise InvalidCanvasPointCloud("PointCloud2 scan duration exceeds one second")
    return maximum_ns


def _decode_raw_lidar_header(metadata: dict[str, Any]) -> tuple[int | None, bool, str]:
    header = metadata.get("lidar_header")
    if not isinstance(header, dict):
        raise InvalidCanvasPointCloud("lidar_header must be an object")
    frame_id = header.get("frame_id")
    if not isinstance(frame_id, str) or not _valid_frame_id(frame_id):
        raise InvalidCanvasPointCloud(f"invalid raw LiDAR frame_id: {frame_id!r}")
    stamp = header.get("stamp")
    if not isinstance(stamp, dict) or not isinstance(stamp.get("valid"), bool):
        raise InvalidCanvasPointCloud("lidar_header.stamp must describe validity")
    valid = stamp["valid"]
    raw_stamp_ns = stamp.get("stamp_ns")
    if valid:
        raw_stamp_ns = _require_int(
            raw_stamp_ns, "lidar_header.stamp.stamp_ns", positive=True
        )
        sec = _require_int(stamp.get("sec"), "lidar_header.stamp.sec")
        nanosec = _require_int(
            stamp.get("nanosec"), "lidar_header.stamp.nanosec"
        )
        if not 0 <= nanosec < 1_000_000_000:
            raise InvalidCanvasPointCloud("lidar_header.stamp.nanosec is out of range")
        if sec * 1_000_000_000 + nanosec != raw_stamp_ns:
            raise InvalidCanvasPointCloud("raw LiDAR stamp fields are inconsistent")
    else:
        raw_stamp_ns = None
    return raw_stamp_ns, valid, frame_id


@dataclass(frozen=True)
class _PointDataTransform:
    applied: bool
    roll_rad: float
    pitch_rad: float
    payload_to_sensor_rotation: np.ndarray
    source_frame_id: str
    attitude_source: str
    attitude_time_correlated: bool


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidCanvasPointCloud(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise InvalidCanvasPointCloud(f"{name} must be finite")
    return number


def _decode_point_data_transform(
    metadata: dict[str, Any],
    *,
    raw_frame_id: str,
) -> _PointDataTransform:
    if metadata.get("point_data_transform") != DRIVER_POINT_DATA_TRANSFORM:
        raise InvalidCanvasPointCloud(
            "unsupported point_data_transform: "
            f"{metadata.get('point_data_transform')!r}"
        )
    params = metadata.get("point_data_transform_params")
    if not isinstance(params, dict):
        raise InvalidCanvasPointCloud(
            "point_data_transform_params are required to restore the rigid LiDAR frame"
        )
    if params.get("version") != POINT_DATA_TRANSFORM_PARAMS_VERSION:
        raise InvalidCanvasPointCloud(
            "unsupported point_data_transform_params version: "
            f"{params.get('version')!r}"
        )
    applied = params.get("applied")
    if not isinstance(applied, bool):
        raise InvalidCanvasPointCloud(
            "point_data_transform_params.applied must be boolean"
        )
    roll_rad = _require_finite_number(
        params.get("roll_rad"), "point_data_transform_params.roll_rad"
    )
    pitch_rad = _require_finite_number(
        params.get("pitch_rad"), "point_data_transform_params.pitch_rad"
    )
    if not -math.pi <= roll_rad <= math.pi:
        raise InvalidCanvasPointCloud(
            "point_data_transform_params.roll_rad is out of range"
        )
    if not -math.pi / 2 <= pitch_rad <= math.pi / 2:
        raise InvalidCanvasPointCloud(
            "point_data_transform_params.pitch_rad is out of range"
        )

    matrix_values = params.get("payload_to_sensor_rotation_row_major")
    if not isinstance(matrix_values, list) or len(matrix_values) != 9:
        raise InvalidCanvasPointCloud(
            "payload_to_sensor_rotation_row_major must contain 9 numbers"
        )
    matrix = np.asarray(
        [
            _require_finite_number(
                value,
                f"payload_to_sensor_rotation_row_major[{index}]",
            )
            for index, value in enumerate(matrix_values)
        ],
        dtype=np.float64,
    ).reshape(3, 3)
    identity = np.eye(3, dtype=np.float64)
    if not np.allclose(matrix @ matrix.T, identity, rtol=0.0, atol=1e-5):
        raise InvalidCanvasPointCloud(
            "payload_to_sensor_rotation_row_major must be orthonormal"
        )
    determinant = float(np.linalg.det(matrix))
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise InvalidCanvasPointCloud(
            "payload_to_sensor_rotation_row_major must be a proper rotation"
        )
    if not applied and not np.allclose(
        matrix, identity, rtol=0.0, atol=1e-7
    ):
        raise InvalidCanvasPointCloud(
            "an unapplied point_data_transform must use the identity rotation"
        )

    source_frame_id = params.get("source_frame_id")
    if source_frame_id != raw_frame_id:
        raise InvalidCanvasPointCloud(
            "point_data_transform source_frame_id must match lidar_header.frame_id"
        )
    attitude_source = params.get("attitude_source")
    if attitude_source != "latest_livox_imu_callback":
        raise InvalidCanvasPointCloud(
            "unsupported point_data_transform attitude_source"
        )
    attitude_time_correlated = params.get("attitude_time_correlated")
    if attitude_time_correlated is not False:
        raise InvalidCanvasPointCloud(
            "point_data_transform attitude_time_correlated must be false"
        )
    return _PointDataTransform(
        applied=applied,
        roll_rad=roll_rad,
        pitch_rad=pitch_rad,
        payload_to_sensor_rotation=matrix,
        source_frame_id=source_frame_id,
        attitude_source=attitude_source,
        attitude_time_correlated=attitude_time_correlated,
    )


def _restore_rigid_lidar_points(
    data: bytes,
    *,
    point_step: int,
    point_count: int,
    transform: _PointDataTransform,
) -> bytes:
    if not transform.applied:
        return data

    restored = bytearray(data)
    records = np.frombuffer(restored, dtype=np.uint8).reshape(
        point_count, point_step
    )
    payload_xyz = (
        records[:, :12]
        .copy()
        .view("<f4")
        .reshape(point_count, 3)
    )
    sensor_xyz = payload_xyz @ transform.payload_to_sensor_rotation.T
    records[:, :12] = np.asarray(sensor_xyz, dtype="<f4").view(np.uint8).reshape(
        point_count, 12
    )
    return bytes(restored)


def decode_canvas_pointcloud(
    payload: bytes | bytearray | memoryview | Iterable[int],
    *,
    receive_stamp_ns: int,
    output_frame_id: str,
    max_source_age_ns: int = DEFAULT_MAX_SOURCE_AGE_NS,
    max_future_skew_ns: int = DEFAULT_MAX_FUTURE_SKEW_NS,
) -> CanvasPointCloud:
    """Decode a legacy-compatible point payload with a strict PCLMETA2 footer.

    The Driver keeps its established gravity-aligned legacy prefix for existing
    dashboard and safety consumers.  PCLMETA2 must carry the exact inverse
    rotation applied to that prefix so this adapter can restore a rigid sensor
    frame before publishing to Nav2.  ``output_frame_id`` is an explicit alias
    for that rigid frame and retains the reviewed static-TF contract.

    Frames without valid v2 metadata or a valid inverse transform are rejected
    instead of silently feeding gravity-aligned points into SLAM.
    """

    try:
        raw = bytes(payload)
    except (TypeError, ValueError) as exc:
        raise InvalidCanvasPointCloud("payload must be a byte sequence") from exc
    if len(raw) < _LEGACY_HEADER.size:
        raise InvalidCanvasPointCloud("payload is shorter than the legacy header")
    if not isinstance(output_frame_id, str) or not _valid_frame_id(output_frame_id):
        raise InvalidCanvasPointCloud(
            f"invalid configured output frame_id: {output_frame_id!r}"
        )

    point_step, point_count = _LEGACY_HEADER.unpack_from(raw)
    _validate_shape(point_step, point_count)
    point_data_end = _LEGACY_HEADER.size + point_step * point_count
    metadata = _decode_metadata(raw, point_data_end=point_data_end)

    if metadata.get("schema") != LIDAR_CLOUD_SCHEMA:
        raise InvalidCanvasPointCloud(f"unsupported LiDAR schema: {metadata.get('schema')!r}")
    if metadata.get("schema_version") != 2:
        raise InvalidCanvasPointCloud(
            f"unsupported LiDAR schema_version: {metadata.get('schema_version')!r}"
        )
    if metadata.get("timestamp_source") != "driver_receive":
        raise InvalidCanvasPointCloud(
            "timestamp_source must be 'driver_receive' for lidar_cloud.v2"
        )
    source_stamp_ns = _require_int(
        metadata.get("source_stamp_ns"), "source_stamp_ns", positive=True
    )
    driver_receive_unix_ns = _require_int(
        metadata.get("driver_receive_unix_ns"),
        "driver_receive_unix_ns",
        positive=True,
    )
    if driver_receive_unix_ns != source_stamp_ns:
        raise InvalidCanvasPointCloud(
            "driver_receive_unix_ns must equal source_stamp_ns"
        )
    driver_receive_monotonic_ns = _require_int(
        metadata.get("driver_receive_monotonic_ns"),
        "driver_receive_monotonic_ns",
        positive=True,
    )
    try:
        validate_source_timestamp_ns(
            source_stamp_ns,
            receive_stamp_ns,
            max_source_age_ns=max_source_age_ns,
            max_future_skew_ns=max_future_skew_ns,
        )
    except InvalidSourceTimestamp as exc:
        raise InvalidCanvasPointCloud(str(exc)) from exc

    raw_stamp_ns, raw_stamp_valid, raw_frame_id = _decode_raw_lidar_header(metadata)
    point_data_transform = _decode_point_data_transform(
        metadata,
        raw_frame_id=raw_frame_id,
    )

    pointcloud = metadata.get("pointcloud")
    if not isinstance(pointcloud, dict):
        raise InvalidCanvasPointCloud("pointcloud metadata must be an object")
    height = _require_int(pointcloud.get("height"), "pointcloud.height", positive=True)
    width = _require_int(pointcloud.get("width"), "pointcloud.width", positive=True)
    metadata_point_step = _require_int(
        pointcloud.get("point_step"), "pointcloud.point_step", positive=True
    )
    row_step = _require_int(
        pointcloud.get("row_step"), "pointcloud.row_step", positive=True
    )
    if height * width != point_count:
        raise InvalidCanvasPointCloud("pointcloud dimensions do not match point_count")
    if metadata_point_step != point_step:
        raise InvalidCanvasPointCloud("pointcloud.point_step does not match legacy prefix")
    if row_step != point_step * width or row_step * height != point_step * point_count:
        raise InvalidCanvasPointCloud("pointcloud row layout is not compact")
    is_bigendian = pointcloud.get("is_bigendian")
    is_dense = pointcloud.get("is_dense")
    if is_bigendian is not False:
        raise InvalidCanvasPointCloud("big-endian point data is unsupported")
    if not isinstance(is_dense, bool):
        raise InvalidCanvasPointCloud("pointcloud.is_dense must be boolean")
    fields = _decode_fields(pointcloud.get("fields"), point_step=point_step)

    source_point_data = raw[_LEGACY_HEADER.size:point_data_end]
    point_data = _restore_rigid_lidar_points(
        source_point_data,
        point_step=point_step,
        point_count=point_count,
        transform=point_data_transform,
    )
    scan_end_offset_ns = point_time_max_offset_ns(
        data=point_data,
        fields=fields,
        height=height,
        width=width,
        point_step=point_step,
        row_step=row_step,
        is_bigendian=False,
    )

    return CanvasPointCloud(
        source_stamp_ns=source_stamp_ns,
        driver_receive_monotonic_ns=driver_receive_monotonic_ns,
        timestamp_source="driver_receive",
        frame_id=output_frame_id,
        frame_source="adapter_alias_for_restored_rigid_lidar_payload",
        source_schema=LIDAR_CLOUD_SCHEMA,
        raw_lidar_stamp_ns=raw_stamp_ns,
        raw_lidar_stamp_valid=raw_stamp_valid,
        raw_lidar_frame_id=raw_frame_id,
        source_point_data_transform=DRIVER_POINT_DATA_TRANSFORM,
        point_data_transform=POINT_DATA_TRANSFORM,
        gravity_alignment_applied=point_data_transform.applied,
        gravity_alignment_roll_rad=point_data_transform.roll_rad,
        gravity_alignment_pitch_rad=point_data_transform.pitch_rad,
        gravity_alignment_attitude_source=point_data_transform.attitude_source,
        gravity_alignment_attitude_time_correlated=(
            point_data_transform.attitude_time_correlated
        ),
        height=height,
        width=width,
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=row_step,
        is_dense=is_dense,
        point_count=point_count,
        scan_end_offset_ns=scan_end_offset_ns,
        data=point_data,
    )


__all__ = [
    "CanvasPointCloud",
    "CanvasPointField",
    "InvalidCanvasPointCloud",
    "LidarClockNormalizer",
    "LidarClockSnapshot",
    "LIDAR_CLOUD_SCHEMA",
    "LIDAR_CLOUD_TRAILER_MAGIC",
    "DRIVER_POINT_DATA_TRANSFORM",
    "NSEC_PER_SEC",
    "POINT_DATA_TRANSFORM",
    "POINT_DATA_TRANSFORM_PARAMS_VERSION",
    "decode_canvas_pointcloud",
    "point_time_max_offset_ns",
]
