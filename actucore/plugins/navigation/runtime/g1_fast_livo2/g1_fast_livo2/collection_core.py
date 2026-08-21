"""ROS-independent configuration and health tracking for FAST-LIVO2 capture."""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path, PurePosixPath
import time

import yaml


COLLECTION_ROOT = PurePosixPath(
    "/opt/phanthy-motus/data/fast_livo2/recordings"
)
COLLECTION_SOURCES = (
    {
        "port": "lidar",
        "topic": "/ubuntu/navigation/lidar",
        "record_topic": "/ubuntu/navigation/collection/lidar",
        "ros_type": "sensor_msgs/msg/PointCloud2",
        "qos": "RELIABLE + KEEP_LAST(depth=10) + VOLATILE",
    },
    {
        "port": "imu",
        "topic": "/ubuntu/navigation/imu",
        "record_topic": "/ubuntu/navigation/collection/imu",
        "ros_type": "sensor_msgs/msg/Imu",
        "qos": "RELIABLE + KEEP_LAST(depth=10) + VOLATILE",
    },
    {
        "port": "rgb_frame",
        "topic": "/ubuntu/camera/rgb_frame",
        "record_topic": "/ubuntu/navigation/collection/camera/rgb",
        "ros_type": "std_msgs/msg/UInt8MultiArray",
        "qos": "RELIABLE + KEEP_LAST(depth=10) + VOLATILE",
    },
    {
        "port": "depth_frame",
        "topic": "/ubuntu/camera/depth_frame",
        "record_topic": "/ubuntu/navigation/collection/camera/depth",
        "ros_type": "std_msgs/msg/UInt8MultiArray",
        "qos": "RELIABLE + KEEP_LAST(depth=10) + VOLATILE",
    },
    {
        "port": "odom",
        "topic": "/ubuntu/navigation/odom",
        "record_topic": "/ubuntu/navigation/collection/odom",
        "ros_type": "nav_msgs/msg/Odometry",
        "qos": "RELIABLE + KEEP_LAST(depth=10) + VOLATILE",
    },
)
ALIGNMENT_SAMPLE_LIMIT = 4096
ALIGNMENT_PAIRS = (
    ("rgb_frame", "depth_frame", 150.0),
    ("lidar", "imu", 20.0),
    ("rgb_frame", "lidar", 60.0),
    ("rgb_frame", "odom", 120.0),
    ("depth_frame", "lidar", 60.0),
)
COLLECTION_SAMPLE_INTERVAL_SEC = 1.0
_RGB_ANCHOR_MATCH_LIMIT_MS = {
    "depth_frame": 150.0,
    "lidar": 60.0,
    "odom": 120.0,
}


def _percentile(values: list[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    index = int(round((len(ordered) - 1) * float(quantile)))
    return ordered[max(0, min(index, len(ordered) - 1))]


def _nearest_skews_ns(first: list[int], second: list[int]) -> list[int]:
    if not first or not second:
        return []
    overlap_start = max(min(first), min(second))
    overlap_end = min(max(first), max(second))
    if overlap_start > overlap_end:
        if overlap_start - overlap_end > 1_000_000_000:
            return []
        candidates = sorted(int(value) for value in first)
        reference = sorted(int(value) for value in second)
    else:
        lower = overlap_start - 1_000_000_000
        upper = overlap_end + 1_000_000_000
        candidates = sorted(
            int(value) for value in first if lower <= value <= upper
        )
        reference = sorted(
            int(value) for value in second if lower <= value <= upper
        )
    if not candidates or not reference:
        return []
    result = []
    cursor = 0
    for value in candidates:
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
        *(item["record_topic"] for item in COLLECTION_SOURCES),
    ]


def read_rosbag_recording_summary(
    directory: str,
    observed_sources: dict,
) -> dict:
    """Reconcile sampled publications with rosbag2's persisted counts."""

    metadata_path = Path(directory) / "metadata.yaml"
    try:
        value = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        information = value["rosbag2_bagfile_information"]
        entries = information["topics_with_message_count"]
        topic_counts = {
            str(entry["topic_metadata"]["name"]): int(entry["message_count"])
            for entry in entries
        }
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        return {
            "healthy": False,
            "failure_reasons": ["recording_metadata_unavailable"],
            "error": f"{type(exc).__name__}: {exc}",
            "topics": {},
        }

    message_count = int(information.get("message_count", 0))
    reasons = ["recording_empty"] if message_count == 0 else []
    topics = {}
    for item in COLLECTION_SOURCES:
        port = item["port"]
        source = observed_sources.get(port, {})
        source_observed = int(source.get("count", 0))
        sampled = int(source.get("sampled_count", source_observed))
        recorded = int(topic_counts.get(item["record_topic"], 0))
        coverage = min(1.0, recorded / sampled) if sampled else 0.0
        topics[port] = {
            "source_topic": item["topic"],
            "record_topic": item["record_topic"],
            "source_observed_count": source_observed,
            "observed_count": sampled,
            "sampled_count": sampled,
            "recorded_count": recorded,
            "recording_coverage": round(coverage, 6),
        }
        if sampled > 0 and coverage < 0.9:
            reasons.append(f"{port}:recording_coverage")
    return {
        "healthy": not reasons,
        "failure_reasons": reasons,
        "error": None,
        "topics": topics,
        "message_count": message_count,
        "duration_ns": int(information.get("duration", {}).get("nanoseconds", 0)),
    }


class CollectionSampler:
    """Select one source-time-aligned multimodal bundle per second."""

    def __init__(self, *, interval_sec: float = COLLECTION_SAMPLE_INTERVAL_SEC):
        self._interval_sec = float(interval_sec)
        self._enabled = False
        self._last_emit_monotonic: float | None = None
        self._emitted_count = 0
        self._rejections: dict[str, int] = {}
        self._last_rejection_reason: str | None = None
        limits = {
            "lidar": 32,
            "imu": 256,
            "rgb_frame": 16,
            "depth_frame": 16,
            "odom": 16,
        }
        self._buffers = {
            item["port"]: deque(maxlen=limits[item["port"]])
            for item in COLLECTION_SOURCES
        }

    def start(self) -> None:
        self._enabled = True
        self._last_emit_monotonic = None
        self._emitted_count = 0
        self._rejections.clear()
        self._last_rejection_reason = None
        for buffer in self._buffers.values():
            buffer.clear()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def stop(self) -> None:
        self._enabled = False
        for buffer in self._buffers.values():
            buffer.clear()

    def snapshot(self) -> dict:
        return {
            "enabled": self._enabled,
            "sample_interval_sec": self._interval_sec,
            "emitted_count": self._emitted_count,
            "rejections": dict(sorted(self._rejections.items())),
            "last_rejection_reason": self._last_rejection_reason,
        }

    def _reject(self, reason: str) -> None:
        self._rejections[reason] = self._rejections.get(reason, 0) + 1
        self._last_rejection_reason = reason
        return None

    def observe(
        self,
        port: str,
        *,
        source_stamp_ns: int | None,
        message: object,
        metadata: dict | None,
        now_monotonic: float | None = None,
    ) -> dict | None:
        if (
            not self._enabled
            or port not in self._buffers
            or source_stamp_ns is None
            or int(source_stamp_ns) <= 0
        ):
            return None
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        sample = {
            "source_stamp_ns": int(source_stamp_ns),
            "message": message,
            "metadata": None if metadata is None else dict(metadata),
        }
        self._buffers[port].append(sample)
        if port != "rgb_frame":
            return None
        if (
            self._last_emit_monotonic is not None
            and now - self._last_emit_monotonic < self._interval_sec
        ):
            return None

        anchor = int(source_stamp_ns)
        bundle = {"rgb_frame": sample}
        for candidate_port, limit_ms in _RGB_ANCHOR_MATCH_LIMIT_MS.items():
            candidates = self._buffers[candidate_port]
            if not candidates:
                return self._reject(f"missing_{candidate_port}")
            nearest = min(
                candidates,
                key=lambda value: abs(int(value["source_stamp_ns"]) - anchor),
            )
            if (
                abs(int(nearest["source_stamp_ns"]) - anchor)
                > int(limit_ms * 1_000_000)
            ):
                return self._reject(f"skew_rgb_frame_{candidate_port}")
            bundle[candidate_port] = nearest
        imu_candidates = self._buffers["imu"]
        if not imu_candidates:
            return self._reject("missing_imu")
        lidar_stamp = int(bundle["lidar"]["source_stamp_ns"])
        bundle["imu"] = min(
            imu_candidates,
            key=lambda value: abs(
                int(value["source_stamp_ns"]) - lidar_stamp
            ),
        )
        for first, second, limit_ms in ALIGNMENT_PAIRS:
            if (
                abs(
                    int(bundle[first]["source_stamp_ns"])
                    - int(bundle[second]["source_stamp_ns"])
                )
                > int(limit_ms * 1_000_000)
            ):
                return self._reject(f"skew_{first}_{second}")
        self._last_emit_monotonic = now
        self._emitted_count += 1
        return bundle


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
        self._sampled_counts = {
            item["port"]: 0 for item in COLLECTION_SOURCES
        }
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
        self._source_errors = {
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
            self._sampled_counts[port] = 0
            self._last_receive_monotonic[port] = None
            self._last_source_stamp_ns[port] = None
            self._source_stamp_counts[port] = 0
            self._source_out_of_order[port] = 0
            self._source_stamps_ns[port].clear()
            self._receive_delays_ns[port].clear()
            self._source_metadata[port] = None
            self._source_errors[port] = None

    def observe_sampled(self, port: str) -> None:
        if not self._enabled or port not in self._sampled_counts:
            return
        self._sampled_counts[port] += 1

    def observe_error(self, port: str, error: str) -> None:
        if not self._enabled or port not in self._source_errors:
            return
        self._source_errors[port] = str(error)
        self._source_metadata[port] = {"decode_error": str(error)}

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
            self._source_errors[port] = None

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
                "sampled_count": int(self._sampled_counts[port]),
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
                "error": self._source_errors[port],
            }

        source_errors = [
            port for port, error in self._source_errors.items() if error
        ]

        alignment_reasons = []
        for port in ("lidar", "imu", "rgb_frame", "depth_frame", "odom"):
            source = sources[port]
            if source["count"] == 0:
                alignment_reasons.append(f"{port}:missing")
            elif source["source_timestamp_coverage"] < 1.0:
                alignment_reasons.append(f"{port}:source_timestamp_coverage")
            if not source["source_monotonic"]:
                alignment_reasons.append(f"{port}:source_timestamp_out_of_order")
            if source["error"]:
                alignment_reasons.append(f"{port}:decode_error")

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
            source_errors = []
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
                failure_reason = (
                    "source_decode_errors:" + ",".join(source_errors)
                    if source_errors
                    else "missing_sources:" + ",".join(missing)
                )
            elif source_errors:
                state = "degraded"
                healthy = False
                failure_reason = "source_decode_errors:" + ",".join(
                    source_errors
                )
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
            "source_errors": source_errors,
            "sources": sources,
            "time_alignment": time_alignment,
        }


__all__ = [
    "COLLECTION_ROOT",
    "COLLECTION_SOURCES",
    "ALIGNMENT_PAIRS",
    "COLLECTION_SAMPLE_INTERVAL_SEC",
    "CollectionSampler",
    "CollectionHealth",
    "finalize_collection_session",
    "normalize_collection_directory",
    "read_rosbag_recording_summary",
    "rosbag_record_command",
]
