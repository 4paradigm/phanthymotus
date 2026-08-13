"""ROS-independent configuration and health tracking for FAST-LIVO2 capture."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import time


COLLECTION_ROOT = PurePosixPath(
    "/opt/phanthy-motus/data/fast_livo2/recordings"
)
COLLECTION_SOURCES = (
    {
        "port": "lidar",
        "topic": "/ubuntu/navigation/lidar_fast_livo",
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
        "port": "rgb",
        "topic": "/ubuntu/camera/rgb",
        "ros_type": "sensor_msgs/msg/CompressedImage",
        "qos": "BEST_EFFORT + KEEP_LAST(depth=4) + VOLATILE",
    },
    {
        "port": "depth",
        "topic": "/ubuntu/camera/depth",
        "ros_type": "sensor_msgs/msg/Image",
        "qos": "BEST_EFFORT + KEEP_LAST(depth=4) + VOLATILE",
    },
    {
        "port": "camera_info",
        "topic": "/ubuntu/camera/camera_info",
        "ros_type": "sensor_msgs/msg/CameraInfo",
        "qos": "BEST_EFFORT + KEEP_LAST(depth=4) + VOLATILE",
    },
)


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

    def observe(
        self,
        port: str,
        *,
        source_stamp_ns: int | None,
        now_monotonic: float | None = None,
    ) -> None:
        if not self._enabled or port not in self._counts:
            return
        self._counts[port] += 1
        self._last_receive_monotonic[port] = (
            time.monotonic() if now_monotonic is None else float(now_monotonic)
        )
        if source_stamp_ns is not None and int(source_stamp_ns) > 0:
            self._last_source_stamp_ns[port] = int(source_stamp_ns)

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
        }


__all__ = [
    "COLLECTION_ROOT",
    "COLLECTION_SOURCES",
    "CollectionHealth",
    "finalize_collection_session",
    "normalize_collection_directory",
    "rosbag_record_command",
]
