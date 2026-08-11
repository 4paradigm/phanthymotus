"""Validate native G1 PointCloud2 data and normalize its LiDAR clock."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import struct
from typing import Iterable

from .timestamp_contract import (
    DEFAULT_MAX_FUTURE_SKEW_NS,
    DEFAULT_MAX_SOURCE_AGE_NS,
    InvalidSourceTimestamp,
    validate_source_timestamp_ns,
)


NSEC_PER_SEC = 1_000_000_000
POINT_FIELD_FLOAT32 = 7
_MIN_POINT_STEP = 12
_MAX_POINT_STEP = 512
_MAX_POINTS = 2_000_000


class InvalidCanvasPointCloud(ValueError):
    """Raised when a native ``sensor_msgs/msg/PointCloud2`` is malformed."""


@dataclass(frozen=True)
class LidarClockSnapshot:
    ready: bool
    mode: str
    samples: int
    offset_ns: int | None
    residual_ns: int | None
    resets: int


class LidarClockNormalizer:
    """Map a native LiDAR clock into the ROS system-time domain.

    If the native header is already close to the ROS receive clock, ``auto``
    mode passes it through unchanged. Otherwise the minimum observed
    ``receive - scan_end`` offset is used so scheduling jitter cannot move the
    LiDAR clock forward. A backwards native stamp starts a new warm-up period.
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
        receive_stamp_ns: int,
        scan_end_offset_ns: int,
    ) -> int | None:
        raw_stamp_ns = int(raw_stamp_ns)
        receive_stamp_ns = int(receive_stamp_ns)
        scan_end_offset_ns = int(scan_end_offset_ns)
        if raw_stamp_ns <= 0 or receive_stamp_ns <= 0:
            raise ValueError("raw and receive timestamps must be positive")
        if not 0 <= scan_end_offset_ns <= NSEC_PER_SEC:
            raise ValueError("scan_end_offset_ns must be within one second")

        if (
            self._last_raw_stamp_ns is not None
            and raw_stamp_ns <= self._last_raw_stamp_ns
        ):
            self._reset()
        self._last_raw_stamp_ns = raw_stamp_ns

        if self._active_mode is None:
            self._active_mode = self._resolve_mode(
                raw_stamp_ns=raw_stamp_ns,
                receive_stamp_ns=receive_stamp_ns,
            )

        if self._active_mode == "passthrough":
            raw_age_ns = receive_stamp_ns - raw_stamp_ns
            if not (
                -self._aligned_tolerance_ns
                <= raw_age_ns
                <= self._aligned_tolerance_ns
            ):
                raise ValueError("passthrough LiDAR clock left system-time domain")
            self._offset_ns = 0
            self._residual_ns = raw_age_ns - scan_end_offset_ns
            return raw_stamp_ns

        candidate = receive_stamp_ns - raw_stamp_ns - scan_end_offset_ns
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

    def _resolve_mode(self, *, raw_stamp_ns: int, receive_stamp_ns: int) -> str:
        if self._configured_mode != "auto":
            return self._configured_mode
        raw_age_ns = receive_stamp_ns - raw_stamp_ns
        if -self._aligned_tolerance_ns <= raw_age_ns <= self._aligned_tolerance_ns:
            return "passthrough"
        return "normalize"

    def _reset(self) -> None:
        self._candidates.clear()
        self._active_mode = None
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


def _validate_shape(point_step: int, point_count: int) -> None:
    if not _MIN_POINT_STEP <= point_step <= _MAX_POINT_STEP:
        raise InvalidCanvasPointCloud(f"invalid point_step: {point_step}")
    if not 0 < point_count <= _MAX_POINTS:
        raise InvalidCanvasPointCloud(f"invalid point_count: {point_count}")


def validate_standard_pointcloud(
    *,
    stamp_sec: int,
    stamp_nanosec: int,
    receive_stamp_ns: int | None,
    frame_id: str,
    height: int,
    width: int,
    point_step: int,
    row_step: int,
    data_length: int,
    field_names: Iterable[str],
    max_source_age_ns: int = DEFAULT_MAX_SOURCE_AGE_NS,
    max_future_skew_ns: int = DEFAULT_MAX_FUTURE_SKEW_NS,
) -> int:
    """Validate native PointCloud2 metadata without rewriting its payload.

    A native LiDAR stamp may use a device clock. Source-age validation is only
    performed when ``receive_stamp_ns`` is provided. The G1 bridge validates
    raw structure first and validates age after clock normalization.
    """
    if isinstance(stamp_sec, bool) or isinstance(stamp_nanosec, bool):
        raise InvalidCanvasPointCloud("PointCloud2 header stamp must be integer")
    stamp_sec = int(stamp_sec)
    stamp_nanosec = int(stamp_nanosec)
    if not 0 <= stamp_nanosec < NSEC_PER_SEC:
        raise InvalidCanvasPointCloud("PointCloud2 header nanosec is out of range")
    source_stamp_ns = stamp_sec * NSEC_PER_SEC + stamp_nanosec
    if source_stamp_ns <= 0:
        raise InvalidCanvasPointCloud("PointCloud2 header stamp must be positive")
    if receive_stamp_ns is not None:
        try:
            validate_source_timestamp_ns(
                source_stamp_ns,
                receive_stamp_ns,
                max_source_age_ns=max_source_age_ns,
                max_future_skew_ns=max_future_skew_ns,
            )
        except InvalidSourceTimestamp as exc:
            raise InvalidCanvasPointCloud(str(exc)) from exc
    if not _valid_frame_id(frame_id):
        raise InvalidCanvasPointCloud(f"invalid ROS frame_id: {frame_id!r}")

    height = int(height)
    width = int(width)
    point_step = int(point_step)
    row_step = int(row_step)
    data_length = int(data_length)
    point_count = height * width
    _validate_shape(point_step, point_count)
    if height < 1 or row_step < point_step * width:
        raise InvalidCanvasPointCloud("invalid PointCloud2 row layout")
    if data_length != row_step * height:
        raise InvalidCanvasPointCloud(
            f"PointCloud2 data size mismatch: expected {row_step * height}, "
            f"got {data_length}"
        )
    names = {str(name) for name in field_names}
    if not {"x", "y", "z"}.issubset(names):
        raise InvalidCanvasPointCloud("PointCloud2 must contain x/y/z fields")
    return source_stamp_ns


def point_time_max_offset_ns(
    *,
    data: bytes | bytearray | memoryview,
    fields: Iterable,
    height: int,
    width: int,
    point_step: int,
    row_step: int,
    is_bigendian: bool,
) -> int:
    """Return the native MID360 scan-end offset from its ``time`` field."""
    time_fields = [field for field in fields if str(field.name) == "time"]
    if len(time_fields) != 1:
        raise InvalidCanvasPointCloud("PointCloud2 requires one time field")
    time_field = time_fields[0]
    offset = int(time_field.offset)
    if (
        int(time_field.datatype) != POINT_FIELD_FLOAT32
        or int(time_field.count) != 1
    ):
        raise InvalidCanvasPointCloud("unexpected PointCloud2 time field layout")
    if offset < 0 or offset + 4 > int(point_step):
        raise InvalidCanvasPointCloud("PointCloud2 time field is outside point_step")

    raw = memoryview(data)
    unpack = struct.Struct(">f" if is_bigendian else "<f").unpack_from
    maximum = 0.0
    try:
        for row in range(int(height)):
            row_offset = row * int(row_step)
            for column in range(int(width)):
                value = float(
                    unpack(raw, row_offset + column * int(point_step) + offset)[0]
                )
                if not math.isfinite(value) or value < 0.0:
                    raise InvalidCanvasPointCloud(
                        "PointCloud2 contains an invalid point time"
                    )
                maximum = max(maximum, value)
    except (struct.error, TypeError, ValueError) as exc:
        raise InvalidCanvasPointCloud("invalid PointCloud2 time buffer") from exc
    maximum_ns = int(round(maximum))
    if maximum_ns > NSEC_PER_SEC:
        raise InvalidCanvasPointCloud("PointCloud2 scan duration exceeds one second")
    return maximum_ns


__all__ = [
    "InvalidCanvasPointCloud",
    "LidarClockNormalizer",
    "LidarClockSnapshot",
    "point_time_max_offset_ns",
    "validate_standard_pointcloud",
]
