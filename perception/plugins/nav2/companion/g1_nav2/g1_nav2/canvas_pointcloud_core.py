"""Decode the timestamped G1 Driver point-cloud envelope for Nav2."""

from __future__ import annotations

from dataclasses import dataclass
import json
import struct
from typing import Any, Iterable

from .timestamp_contract import (
    DEFAULT_MAX_FUTURE_SKEW_NS,
    DEFAULT_MAX_SOURCE_AGE_NS,
    InvalidSourceTimestamp,
    validate_source_timestamp_ns,
)


LIDAR_CLOUD_SCHEMA = "phanthy.g1.lidar_cloud.v2"
LIDAR_CLOUD_TRAILER_MAGIC = b"PCLMETA2"
POINT_DATA_TRANSFORM = "gravity_aligned_roll_pitch"
POINT_FIELD_FLOAT32 = 7
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
    point_data_transform: str
    height: int
    width: int
    fields: tuple[CanvasPointField, ...]
    is_bigendian: bool
    point_step: int
    row_step: int
    is_dense: bool
    point_count: int
    data: bytes


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


def decode_canvas_pointcloud(
    payload: bytes | bytearray | memoryview | Iterable[int],
    *,
    receive_stamp_ns: int,
    output_frame_id: str,
    max_source_age_ns: int = DEFAULT_MAX_SOURCE_AGE_NS,
    max_future_skew_ns: int = DEFAULT_MAX_FUTURE_SKEW_NS,
) -> CanvasPointCloud:
    """Decode a legacy-compatible point payload with a strict PCLMETA2 footer.

    The Driver rotates xyz into its established gravity-aligned payload before
    publishing. Therefore the raw ``lidar_header.frame_id`` is diagnostic only;
    ``output_frame_id`` remains the reviewed adapter frame used by Nav2.
    Frames without valid v2 metadata are rejected instead of receiving a
    fabricated adapter timestamp.
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

    if metadata.get("point_data_transform") != POINT_DATA_TRANSFORM:
        raise InvalidCanvasPointCloud(
            "unsupported point_data_transform: "
            f"{metadata.get('point_data_transform')!r}"
        )
    raw_stamp_ns, raw_stamp_valid, raw_frame_id = _decode_raw_lidar_header(metadata)

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

    return CanvasPointCloud(
        source_stamp_ns=source_stamp_ns,
        driver_receive_monotonic_ns=driver_receive_monotonic_ns,
        timestamp_source="driver_receive",
        frame_id=output_frame_id,
        frame_source="adapter_contract_for_gravity_aligned_payload",
        source_schema=LIDAR_CLOUD_SCHEMA,
        raw_lidar_stamp_ns=raw_stamp_ns,
        raw_lidar_stamp_valid=raw_stamp_valid,
        raw_lidar_frame_id=raw_frame_id,
        point_data_transform=POINT_DATA_TRANSFORM,
        height=height,
        width=width,
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=row_step,
        is_dense=is_dense,
        point_count=point_count,
        data=raw[_LEGACY_HEADER.size:point_data_end],
    )


__all__ = [
    "CanvasPointCloud",
    "CanvasPointField",
    "InvalidCanvasPointCloud",
    "LIDAR_CLOUD_SCHEMA",
    "LIDAR_CLOUD_TRAILER_MAGIC",
    "POINT_DATA_TRANSFORM",
    "decode_canvas_pointcloud",
]
