"""Vectorized PointCloud2 operations for the FAST-LIVO2 adapter hot path."""

from __future__ import annotations

import math
import struct
from typing import Iterable

import numpy as np

from .frame_adapter_core import (
    FLOAT32,
    FLOAT64,
    InvalidFastLivo2Frame,
    Pose3,
    normalize_quaternion,
    yaw_from_quaternion,
)


_FLOAT32_MAX = float(np.finfo(np.float32).max)
_MAP_VIEW_HEADER_SIZE = struct.calcsize("<fffBI")


def _field_value(field, key: str):
    return field[key] if isinstance(field, dict) else getattr(field, key)


def _validated_xyz_array(points, *, context: str) -> np.ndarray:
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidFastLivo2Frame(f"{context} coordinates are invalid") from exc
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise InvalidFastLivo2Frame(f"{context} must have shape (N, 3)")
    if not np.isfinite(array).all():
        raise InvalidFastLivo2Frame(f"{context} must contain finite coordinates")
    if array.size and np.max(np.abs(array)) > _FLOAT32_MAX:
        raise InvalidFastLivo2Frame(
            f"{context} exceeds the float32 coordinate range"
        )
    return array


def decode_xyz_array(
    *,
    fields: Iterable,
    data: bytes,
    point_step: int,
    row_step: int,
    width: int,
    height: int,
    is_bigendian: bool,
    max_points: int | None = None,
    max_data_bytes: int | None = None,
) -> np.ndarray:
    """Decode finite XYZ rows from PointCloud2 without a Python point loop."""

    if is_bigendian:
        raise InvalidFastLivo2Frame("big-endian PointCloud2 is unsupported")
    if point_step <= 0 or row_step < 0 or width < 0 or height < 0:
        raise InvalidFastLivo2Frame("invalid PointCloud2 dimensions")
    if max_points is not None and max_points < 1:
        raise ValueError("max_points must be positive when provided")
    if max_data_bytes is not None and max_data_bytes < 1:
        raise ValueError("max_data_bytes must be positive when provided")

    layout: dict[str, tuple[int, int]] = {}
    for field in fields:
        name = str(_field_value(field, "name"))
        if name not in {"x", "y", "z"}:
            continue
        datatype = int(_field_value(field, "datatype"))
        if (
            int(_field_value(field, "count")) != 1
            or datatype not in {FLOAT32, FLOAT64}
        ):
            raise InvalidFastLivo2Frame(
                f"{name} must be scalar float32/float64"
            )
        offset = int(_field_value(field, "offset"))
        size = 4 if datatype == FLOAT32 else 8
        if offset < 0 or offset + size > point_step:
            raise InvalidFastLivo2Frame(f"{name} field exceeds point_step")
        layout[name] = (offset, datatype)
    if set(layout) != {"x", "y", "z"}:
        raise InvalidFastLivo2Frame("PointCloud2 requires x/y/z fields")

    point_count = width * height
    if max_points is not None and point_count > max_points:
        raise InvalidFastLivo2Frame(
            f"PointCloud2 exceeds {max_points} point safety limit"
        )
    if max_data_bytes is not None and len(data) > max_data_bytes:
        raise InvalidFastLivo2Frame(
            f"PointCloud2 exceeds {max_data_bytes} byte safety limit"
        )
    if point_count == 0:
        return np.empty((0, 3), dtype=np.float32)

    effective_row_step = row_step or width * point_step
    if effective_row_step < width * point_step:
        raise InvalidFastLivo2Frame("PointCloud2 row_step is too small")
    required_bytes = (height - 1) * effective_row_step + width * point_step
    if len(data) < required_bytes:
        raise InvalidFastLivo2Frame("PointCloud2 data is truncated")

    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [
                "<f4" if layout[name][1] == FLOAT32 else "<f8"
                for name in ("x", "y", "z")
            ],
            "offsets": [layout[name][0] for name in ("x", "y", "z")],
            "itemsize": point_step,
        }
    )
    try:
        structured = np.ndarray(
            shape=(height, width),
            dtype=dtype,
            buffer=data,
            strides=(effective_row_step, point_step),
        )
    except (TypeError, ValueError, BufferError) as exc:
        raise InvalidFastLivo2Frame("PointCloud2 layout is invalid") from exc

    xyz = np.column_stack(
        (
            structured["x"].reshape(-1),
            structured["y"].reshape(-1),
            structured["z"].reshape(-1),
        )
    ).astype(np.float64, copy=False)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    if xyz.size and np.max(np.abs(xyz)) > _FLOAT32_MAX:
        raise InvalidFastLivo2Frame(
            "PointCloud2 point exceeds the float32 coordinate range"
        )
    return np.ascontiguousarray(xyz, dtype=np.float32)


def absolute_point_time_span_ms(
    *,
    fields: Iterable,
    data: bytes,
    point_step: int,
    row_step: int,
    width: int,
    height: int,
    is_bigendian: bool,
    header_stamp_ns: int,
    max_scan_span_ms: float = 200.0,
) -> float:
    """Validate the navigation cloud contract and return its point-time span."""

    if is_bigendian:
        raise InvalidFastLivo2Frame("big-endian PointCloud2 is unsupported")
    if point_step <= 0 or row_step < 0 or width < 0 or height < 0:
        raise InvalidFastLivo2Frame("invalid PointCloud2 dimensions")
    if header_stamp_ns <= 0:
        raise InvalidFastLivo2Frame("PointCloud2 header stamp must be positive")
    layout = {
        str(_field_value(field, "name")): field
        for field in fields
        if str(_field_value(field, "name")) in {"x", "y", "z", "timestamp"}
    }
    if set(layout) != {"x", "y", "z", "timestamp"}:
        raise InvalidFastLivo2Frame(
            "navigation PointCloud2 requires x/y/z/timestamp fields"
        )
    for name in ("x", "y", "z"):
        field = layout[name]
        if int(_field_value(field, "count")) != 1 or int(
            _field_value(field, "datatype")
        ) not in {FLOAT32, FLOAT64}:
            raise InvalidFastLivo2Frame(
                f"{name} must be scalar float32/float64"
            )
    timestamp = layout["timestamp"]
    if int(_field_value(timestamp, "count")) != 1 or int(
        _field_value(timestamp, "datatype")
    ) != FLOAT64:
        raise InvalidFastLivo2Frame("timestamp must be scalar float64 absolute ns")
    offset = int(_field_value(timestamp, "offset"))
    if offset < 0 or offset + 8 > point_step:
        raise InvalidFastLivo2Frame("timestamp field exceeds point_step")

    point_count = width * height
    if point_count < 2:
        raise InvalidFastLivo2Frame("point timestamps require at least two points")
    effective_row_step = row_step or width * point_step
    if effective_row_step < width * point_step:
        raise InvalidFastLivo2Frame("PointCloud2 row_step is too small")
    required_bytes = (height - 1) * effective_row_step + width * point_step
    if len(data) < required_bytes:
        raise InvalidFastLivo2Frame("PointCloud2 data is truncated")
    try:
        values = np.ndarray(
            shape=(height, width),
            dtype=np.dtype(
                {
                    "names": ["timestamp"],
                    "formats": ["<f8"],
                    "offsets": [offset],
                    "itemsize": point_step,
                }
            ),
            buffer=data,
            strides=(effective_row_step, point_step),
        )["timestamp"].reshape(-1)
    except (TypeError, ValueError, BufferError) as exc:
        raise InvalidFastLivo2Frame("PointCloud2 layout is invalid") from exc
    if not np.isfinite(values).all():
        raise InvalidFastLivo2Frame("point timestamps must be finite")
    if np.any(np.diff(values) < 0):
        raise InvalidFastLivo2Frame("point timestamps must be monotonic")
    first = float(values[0])
    last = float(values[-1])
    maximum = float(header_stamp_ns) + float(max_scan_span_ms) * 1_000_000.0
    if first < float(header_stamp_ns) or last > maximum:
        raise InvalidFastLivo2Frame(
            "point timestamps must be within header stamp + scan span"
        )
    span_ms = (last - first) / 1_000_000.0
    if not 0.0 < span_ms <= float(max_scan_span_ms):
        raise InvalidFastLivo2Frame("point timestamp span must be within (0, 200] ms")
    return span_ms


def transform_xyz_array(pose: Pose3, points) -> np.ndarray:
    """Apply pose to an ``(N, 3)`` cloud with NumPy matrix operations."""

    source = _validated_xyz_array(points, context="transform input point")
    translation = np.asarray((pose.x, pose.y, pose.z), dtype=np.float64)
    if not np.isfinite(translation).all() or np.max(np.abs(translation)) > _FLOAT32_MAX:
        raise InvalidFastLivo2Frame("transform translation is invalid")
    q = normalize_quaternion(pose.q)
    xx, yy, zz = q.x * q.x, q.y * q.y, q.z * q.z
    xy, xz, yz = q.x * q.y, q.x * q.z, q.y * q.z
    wx, wy, wz = q.w * q.x, q.w * q.y, q.w * q.z
    rotation = np.asarray(
        (
            (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
            (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
            (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
        ),
        dtype=np.float64,
    )
    transformed = source @ rotation.T + translation
    if not np.isfinite(transformed).all():
        raise InvalidFastLivo2Frame("transformed point is not finite")
    if transformed.size and np.max(np.abs(transformed)) > _FLOAT32_MAX:
        raise InvalidFastLivo2Frame(
            "transformed point exceeds the float32 coordinate range"
        )
    return np.ascontiguousarray(transformed, dtype=np.float32)


def xyz_array_bytes(points) -> bytes:
    """Encode validated XYZ rows as packed little-endian float32 bytes."""

    array = _validated_xyz_array(points, context="output point")
    return np.ascontiguousarray(array, dtype="<f4").tobytes()


def map_view_with_pose(frame: bytes, pose: Pose3) -> bytes:
    """Replace only the pose prefix of an already encoded map-view frame."""

    if len(frame) < _MAP_VIEW_HEADER_SIZE:
        raise InvalidFastLivo2Frame("cached map-view frame is truncated")
    values = (pose.x, pose.y, yaw_from_quaternion(pose.q))
    if not all(math.isfinite(value) and abs(value) <= _FLOAT32_MAX for value in values):
        raise InvalidFastLivo2Frame("map-view pose is outside float32 range")
    try:
        pose_prefix = struct.pack("<fff", *values)
    except (OverflowError, struct.error) as exc:
        raise InvalidFastLivo2Frame("map-view pose cannot be encoded") from exc
    return pose_prefix + frame[12:]
