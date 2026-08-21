"""ROS-independent configuration and health tracking for FAST-LIVO2 capture."""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path, PurePosixPath
import time


COLLECTION_ROOT = PurePosixPath(
    "/opt/phanthy-motus/data/fast_livo2/recordings"
)
COLLECTION_SOURCES = (
    {
        "port": "lidar",
        "topic": "/ubuntu/navigation/lidar",
        "ros_type": "sensor_msgs/msg/PointCloud2",
        "qos": "RELIABLE + KEEP_LAST(depth=2) + VOLATILE",
    },
    {
        "port": "imu",
        "topic": "/ubuntu/navigation/imu",
        "ros_type": "sensor_msgs/msg/Imu",
        "qos": "RELIABLE + KEEP_LAST(depth=200) + VOLATILE",
    },
    {
        "port": "rgb_v2",
        "topic": "/ubuntu/navigation/camera/rgb",
        "ros_type": "std_msgs/msg/UInt8MultiArray",
        "qos": "BEST_EFFORT + KEEP_LAST(depth=4) + VOLATILE",
    },
    {
        "port": "depth",
        "topic": "/ubuntu/camera/depth",
        "ros_type": "sensor_msgs/msg/Image",
        "qos": "BEST_EFFORT + KEEP_LAST(depth=4) + VOLATILE",
    },
    {
        "port": "odom",
        "topic": "/ubuntu/navigation/odom",
        "ros_type": "nav_msgs/msg/Odometry",
        "qos": "BEST_EFFORT + KEEP_LAST(depth=20) + VOLATILE",
    },
)
ALIGNMENT_SAMPLE_LIMIT = 4096
ALIGNMENT_PAIRS = (
    ("rgb_v2", "depth", 20.0),
    ("lidar", "imu", 20.0),
    ("rgb_v2", "lidar", 20.0),
    ("rgb_v2", "odom", 20.0),
    ("depth", "lidar", 20.0),
)


def _percentile(values: list[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    index = int(round((len(ordered) - 1) * float(quantile)))
    return ordered[max(0, min(index, len(ordered) - 1))]


def _nearest_skews_ns(first: list[int], second: list[int]) -> list[int]:
    if not first or not second:
        return []
    reference = sorted(int(value) for value in second)
    result = []
    cursor = 0
    for value in sorted(int(item) for item in first):
        while (
            cursor + 1 < len(reference)
            and abs(reference[cursor + 1] - value)
            <= abs(reference[cursor] - value)
        ):
            cursor += 1
        result.append(abs(reference[cursor] - value))
    return result


def _ns_to_ms(value: int | None) -> float | None:
    if value is None:
        return None
    return round(int(value) / 1_000_000, 3)


def normalize_collection_directory(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("collection_directory must be a non-empty absolute path")
    path = PurePosixPath(value.strip())
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("collection_directory must be a safe absolute path")
    try:
        path.relative_to(COLLECTION_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"collection_directory must be within {COLLECTION_ROOT}"
        ) from exc
    return str(path)


def rosbag_record_command(output_directory: str) -> list[str]:
    return [
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--output",
        output_directory,
        *(item["topic"] for item in COLLECTION_SOURCES),
    ]


def finalize_collection_session(
    partial_directory: str,
    final_directory: str,
    receipt: dict,
    *,
    storage_complete: bool,
) -> dict:
    partial = Path(partial_directory)
    final = Path(final_directory)
    if not partial.is_dir():
        raise FileNotFoundError(f"collection partial directory is missing: {partial}")
    result = dict(receipt)
    result["directory"] = str(final if storage_complete else partial)
    temporary = partial / "collection.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(partial / "collection.json")
    if storage_complete:
        if final.exists():
            raise FileExistsError(f"collection final directory already exists: {final}")
        partial.replace(final)
    return result


class CollectionHealth:
    """Track source arrival without doing file I/O in ROS callbacks."""

    def __init__(
        self,
        *,
        grace_sec: float = 5.0,
        stale_sec: float = 2.0,
    ) -> None:
        self._grace_sec = float(grace_sec)
        self._stale_sec = float(stale_sec)
        self._enabled = False
        self._started_monotonic: float | None = None
        self._session_id: str | None = None
        self._directory: str | None = None
        self._counts = {item["port"]: 0 for item in COLLECTION_SOURCES}
        self._last_receive_monotonic = {
            item["port"]: None for item in COLLECTION_SOURCES
        }
        self._last_source_stamp_ns = {
            item["port"]: None for item in COLLECTION_SOURCES
        }
        self._source_stamp_counts = {
            item["port"]: 0 for item in COLLECTION_SOURCES
        }
        self._source_out_of_order = {
            item["port"]: 0 for item in COLLECTION_SOURCES
        }
        self._source_stamps_ns = {
            item["port"]: deque(maxlen=ALIGNMENT_SAMPLE_LIMIT)
            for item in COLLECTION_SOURCES
        }
        self._receive_delays_ns = {
            item["port"]: deque(maxlen=ALIGNMENT_SAMPLE_LIMIT)
            for item in COLLECTION_SOURCES
        }
        self._source_metadata = {
            item["port"]: None for item in COLLECTION_SOURCES
        }

    def start(
        self,
        session_id: str,
        directory: str,
        *,
        now_monotonic: float | None = None,
    ) -> None:
        self._enabled = True
        self._started_monotonic = (
            time.monotonic() if now_monotonic is None else float(now_monotonic)
        )
        self._session_id = str(session_id)
        self._directory = str(directory)
        for port in self._counts:
            self._counts[port] = 0
            self._last_receive_monotonic[port] = None
            self._last_source_stamp_ns[port] = None
            self._source_stamp_counts[port] = 0
            self._source_out_of_order[port] = 0
            self._source_stamps_ns[port].clear()
            self._receive_delays_ns[port].clear()
            self._source_metadata[port] = None

    def observe(
        self,
        port: str,
        *,
        source_stamp_ns: int | None,
        now_monotonic: float | None = None,
        receive_epoch_ns: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        if not self._enabled or port not in self._counts:
            return
        self._counts[port] += 1
        self._last_receive_monotonic[port] = (
            time.monotonic() if now_monotonic is None else float(now_monotonic)
        )
        if source_stamp_ns is not None and int(source_stamp_ns) > 0:
            stamp_ns = int(source_stamp_ns)
            previous = self._last_source_stamp_ns[port]
            if previous is not None and stamp_ns < previous:
                self._source_out_of_order[port] += 1
            self._last_source_stamp_ns[port] = stamp_ns
            self._source_stamp_counts[port] += 1
            self._source_stamps_ns[port].append(stamp_ns)
            received_ns = (
                time.time_ns()
                if receive_epoch_ns is None
                else int(receive_epoch_ns)
            )
            self._receive_delays_ns[port].append(received_ns - stamp_ns)
        if isinstance(metadata, dict):
            self._source_metadata[port] = dict(metadata)

    def stop(self) -> None:
        self._enabled = False

    def snapshot(
        self,
        *,
        process_running: bool,
        process_return_code: int | None = None,
        process_error: str | None = None,
        now_monotonic: float | None = None,
    ) -> dict:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        sources = {}
        missing = []
        stale = []
        for item in COLLECTION_SOURCES:
            port = item["port"]
            last = self._last_receive_monotonic[port]
            count = int(self._counts[port])
            if count == 0:
                missing.append(port)
            elif last is not None and now - last > self._stale_sec:
                stale.append(port)
            sources[port] = {
                **item,
                "count": count,
                "last_receive_age_sec": (
                    None if last is None else round(max(0.0, now - last), 3)
                ),
                "last_source_stamp_ns": self._last_source_stamp_ns[port],
                "source_timestamp_count": self._source_stamp_counts[port],
                "source_timestamp_coverage": round(
                    self._source_stamp_counts[port] / count if count else 0.0,
                    6,
                ),
                "source_monotonic": self._source_out_of_order[port] == 0,
                "source_out_of_order_timestamps": self._source_out_of_order[port],
                "receive_delay_ms": {
                    "p50": _ns_to_ms(
                        _percentile(list(self._receive_delays_ns[port]), 0.50)
                    ),
                    "p95": _ns_to_ms(
                        _percentile(list(self._receive_delays_ns[port]), 0.95)
                    ),
                    "max": _ns_to_ms(
                        max(self._receive_delays_ns[port])
                        if self._receive_delays_ns[port]
                        else None
                    ),
                },
                "metadata": (
                    None
                    if self._source_metadata[port] is None
                    else dict(self._source_metadata[port])
                ),
            }

        alignment_reasons = []
        for port in ("lidar", "imu", "rgb_v2", "depth", "odom"):
            source = sources[port]
            if source["count"] == 0:
                alignment_reasons.append(f"{port}:missing")
            elif source["source_timestamp_coverage"] < 1.0:
                alignment_reasons.append(f"{port}:source_timestamp_coverage")
            if not source["source_monotonic"]:
                alignment_reasons.append(f"{port}:source_timestamp_out_of_order")

        pairs = {}
        for first, second, threshold_ms in ALIGNMENT_PAIRS:
            skews = _nearest_skews_ns(
                list(self._source_stamps_ns[first]),
                list(self._source_stamps_ns[second]),
            )
            p95_ns = _percentile(skews, 0.95)
            p95_ms = _ns_to_ms(p95_ns)
            ready = bool(skews) and p95_ms is not None and p95_ms <= threshold_ms
            name = f"{first}_{second}"
            pairs[name] = {
                "samples": len(skews),
                "nearest_skew_ms": {
                    "p50": _ns_to_ms(_percentile(skews, 0.50)),
                    "p95": p95_ms,
                    "max": _ns_to_ms(max(skews) if skews else None),
                },
                "p95_limit_ms": threshold_ms,
                "ready": ready,
            }
            if not ready:
                alignment_reasons.append(f"{name}:nearest_skew")

        time_alignment = {
            "level": "software_time_aligned",
            "clock_domain": "ros_system_time",
            "hardware_synchronized": False,
            "alignment_ready": not alignment_reasons,
            "pairs": pairs,
            "reasons": alignment_reasons,
            "note": (
                "Source headers are compared in the ROS system-time domain; "
                "this does not prove PTP or hardware-trigger synchronization."
            ),
        }

        if not self._enabled:
            state = "disabled"
            healthy = True
            failure_reason = None
            missing = []
            stale = []
        elif process_error:
            state = "error"
            healthy = False
            failure_reason = process_error
        elif not process_running:
            state = "error"
            healthy = False
            failure_reason = f"rosbag_exited:{process_return_code}"
        else:
            elapsed = max(0.0, now - float(self._started_monotonic or now))
            if missing and elapsed < self._grace_sec:
                state = "starting"
                healthy = False
                failure_reason = None
            elif missing:
                state = "degraded"
                healthy = False
                failure_reason = "missing_sources:" + ",".join(missing)
            elif stale:
                state = "degraded"
                healthy = False
                failure_reason = "stale_sources:" + ",".join(stale)
            elif alignment_reasons and elapsed < self._grace_sec:
                state = "starting"
                healthy = False
                failure_reason = None
            elif alignment_reasons:
                state = "degraded"
                healthy = False
                failure_reason = "timestamp_alignment:" + ",".join(
                    alignment_reasons
                )
            else:
                state = "recording"
                healthy = True
                failure_reason = None

        return {
            "schema": "phanthy.navigation.fast_livo2_collection_status.v1",
            "enabled": self._enabled,
            "state": state,
            "healthy": healthy,
            "failure_reason": failure_reason,
            "session_id": self._session_id,
            "directory": self._directory,
            "missing_sources": missing,
            "stale_sources": stale,
            "sources": sources,
            "time_alignment": time_alignment,
        }


__all__ = [
    "COLLECTION_ROOT",
    "COLLECTION_SOURCES",
    "ALIGNMENT_PAIRS",
    "CollectionHealth",
    "finalize_collection_session",
    "normalize_collection_directory",
    "rosbag_record_command",
]
