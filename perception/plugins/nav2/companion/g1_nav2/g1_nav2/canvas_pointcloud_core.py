"""Decode released and timestamped G1 canvas point-cloud envelopes."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable


_MAGIC = b"PCV2"
_VERSION = 2
_HEADER = struct.Struct("<4sHHIIqH")
_LEGACY_HEADER = struct.Struct("<II")
_MIN_POINT_STEP = 12
_MAX_POINT_STEP = 512
_MAX_POINTS = 2_000_000
_MAX_FRAME_ID_BYTES = 128


class InvalidCanvasPointCloud(ValueError):
    """Raised when a ``sensor/pointcloud`` envelope is malformed."""


@dataclass(frozen=True)
class CanvasPointCloud:
    source_stamp_ns: int
    frame_id: str
    timestamp_source: str
    frame_source: str
    source_schema: str
    point_step: int
    point_count: int
    data: bytes


def _valid_frame_id(frame_id: str) -> bool:
    return bool(
        frame_id
        and not frame_id.startswith("/")
        and "//" not in frame_id
        and frame_id.replace("/", "").replace("_", "").isalnum()
    )


def _validate_shape(point_step: int, point_count: int) -> None:
    if not _MIN_POINT_STEP <= point_step <= _MAX_POINT_STEP:
        raise InvalidCanvasPointCloud(f"invalid point_step: {point_step}")
    if not 0 < point_count <= _MAX_POINTS:
        raise InvalidCanvasPointCloud(f"invalid point_count: {point_count}")


def decode_canvas_pointcloud(
    payload: bytes | bytearray | memoryview | Iterable[int],
    *,
    receive_stamp_ns: int | None = None,
    legacy_frame_id: str | None = None,
) -> CanvasPointCloud:
    """Decode the released legacy envelope or timestamped v2 envelope.

    Header layout is ``magic, version, flags, point_step, point_count,
    source_stamp_ns, frame_id_length`` followed by UTF-8 ``frame_id`` and raw
    point bytes. x/y/z are float32 metres at offsets 0/4/8.

    The released G1 Driver currently emits ``point_step, point_count, data``.
    It carries neither stamp nor frame, so callers must provide an adapter
    receive timestamp and the configured MID360 frame. These are explicitly
    labelled as adapter metadata, not Driver source metadata.
    """

    try:
        raw = bytes(payload)
    except (TypeError, ValueError) as exc:
        raise InvalidCanvasPointCloud("payload must be a byte sequence") from exc
    if raw[:4] != _MAGIC:
        if len(raw) < _LEGACY_HEADER.size:
            raise InvalidCanvasPointCloud(
                f"payload is shorter than the {_LEGACY_HEADER.size}-byte legacy header"
            )
        point_step, point_count = _LEGACY_HEADER.unpack_from(raw)
        _validate_shape(point_step, point_count)
        expected_size = _LEGACY_HEADER.size + point_step * point_count
        if len(raw) != expected_size:
            raise InvalidCanvasPointCloud(
                f"payload size mismatch: expected {expected_size}, got {len(raw)}"
            )
        if isinstance(receive_stamp_ns, bool) or not isinstance(receive_stamp_ns, int):
            raise InvalidCanvasPointCloud(
                "legacy point cloud requires an adapter receive timestamp"
            )
        if receive_stamp_ns <= 0:
            raise InvalidCanvasPointCloud("receive_stamp_ns must be positive")
        if not isinstance(legacy_frame_id, str) or not _valid_frame_id(legacy_frame_id):
            raise InvalidCanvasPointCloud(
                f"invalid configured legacy frame_id: {legacy_frame_id!r}"
            )
        return CanvasPointCloud(
            source_stamp_ns=receive_stamp_ns,
            frame_id=legacy_frame_id,
            timestamp_source="adapter_receive",
            frame_source="adapter_contract",
            source_schema="unitree.g1.pointcloud.legacy",
            point_step=point_step,
            point_count=point_count,
            data=raw[_LEGACY_HEADER.size:],
        )

    if len(raw) < _HEADER.size:
        raise InvalidCanvasPointCloud(
            f"payload is shorter than the {_HEADER.size}-byte v2 header"
        )

    (
        magic,
        version,
        flags,
        point_step,
        point_count,
        source_stamp_ns,
        frame_id_length,
    ) = _HEADER.unpack_from(raw)
    if magic != _MAGIC or version != _VERSION:
        raise InvalidCanvasPointCloud(
            f"unsupported point-cloud envelope version: {version}"
        )
    if flags != 0:
        raise InvalidCanvasPointCloud(f"unsupported point-cloud flags: {flags}")
    if source_stamp_ns <= 0:
        raise InvalidCanvasPointCloud("source_stamp_ns must be positive")
    if not 0 < frame_id_length <= _MAX_FRAME_ID_BYTES:
        raise InvalidCanvasPointCloud(
            f"invalid frame_id length: {frame_id_length}"
        )
    _validate_shape(point_step, point_count)

    data_offset = _HEADER.size + frame_id_length
    expected_size = data_offset + point_step * point_count
    if len(raw) != expected_size:
        raise InvalidCanvasPointCloud(
            f"payload size mismatch: expected {expected_size}, got {len(raw)}"
        )
    try:
        frame_id = raw[_HEADER.size:data_offset].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidCanvasPointCloud("frame_id is not valid UTF-8") from exc
    if not _valid_frame_id(frame_id):
        raise InvalidCanvasPointCloud(f"invalid ROS frame_id: {frame_id!r}")
    return CanvasPointCloud(
        source_stamp_ns=source_stamp_ns,
        frame_id=frame_id,
        timestamp_source="driver",
        frame_source="driver_payload",
        source_schema="phanthy.sensor.pointcloud.v2",
        point_step=point_step,
        point_count=point_count,
        data=raw[data_offset:],
    )


__all__ = [
    "CanvasPointCloud",
    "InvalidCanvasPointCloud",
    "decode_canvas_pointcloud",
]
