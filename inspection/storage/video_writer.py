from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .atomic_writer import _sha256, _write_json_temp, _write_open_state, fsync_directory
from .ledger import SegmentLedger
from .layout import card_storage_slug, instance_storage_slug, segment_basename, utc_hour_partition
from .models import SegmentRecord


class VideoFragmentStore:
    """Owns splitmuxsink locations and atomically commits closed MP4 fragments."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        ledger: SegmentLedger,
        card_id: str,
        instance_id: str,
        input_topic: str,
        session_id: str,
        device_id: str,
        encoder: str,
        target_bitrate_kbps: int,
    ) -> None:
        self.data_root = Path(data_root)
        self.ledger = ledger
        self.card_id = card_id
        self.instance_id = instance_id
        self.input_topic = input_topic
        self.session_id = session_id
        self.device_id = device_id
        self.storage_card_slug = card_storage_slug(card_id)
        self.storage_instance_slug = instance_storage_slug(instance_id, input_topic)
        self.encoder = encoder
        self.target_bitrate_kbps = int(target_bitrate_kbps)
        self._lock = threading.RLock()
        self._sequence = 0
        self._open: dict[str, dict[str, Any]] = {}
        self._current_location = ""
        self.finalized_segments = 0
        self.last_finalized: dict[str, Any] | None = None

    def create_location(self, fragment_id: int, *, first_source_stamp_ns: int = 0) -> str:
        wall_start_ns = time.time_ns()
        utc = datetime.fromtimestamp(wall_start_ns / 1_000_000_000, tz=timezone.utc)
        directory = self.data_root / self.storage_card_slug / self.storage_instance_slug / utc_hour_partition(wall_start_ns)
        directory.mkdir(parents=True, exist_ok=True)
        sequence = self._sequence
        self._sequence += 1
        part_path = directory / f"{segment_basename(wall_start_ns, sequence, 'mp4')}.part"
        open_info = {
            "schema_version": "1.0",
            "segment_id": f"seg-{uuid.uuid4().hex}",
            "kind": "video",
            "device_id": self.device_id,
            "card_id": self.card_id,
            "instance_id": self.instance_id,
            "storage_card_slug": self.storage_card_slug,
            "storage_instance_slug": self.storage_instance_slug,
            "storage_time_partition": utc_hour_partition(wall_start_ns),
            "input_topic": self.input_topic,
            "format": "video/h264",
            "source_format": "image/jpeg",
            "ros_type": "sensor_msgs/msg/CompressedImage",
            "session_id": self.session_id,
            "sequence": sequence,
            "gstreamer_fragment_id": int(fragment_id),
            "encoder": self.encoder,
            "target_bitrate_kbps": self.target_bitrate_kbps,
            "wall_clock_start_ns": wall_start_ns,
            "wall_clock_start_utc": utc.isoformat().replace("+00:00", "Z"),
            "receive_monotonic_start_ns": time.monotonic_ns(),
            "receive_monotonic_end_ns": 0,
            "source_stamp_start_ns": int(first_source_stamp_ns),
            "source_stamp_end_ns": int(first_source_stamp_ns),
            "source_stamp_valid": int(first_source_stamp_ns) > 0,
            "samples_or_frames": 0,
            "dropped_before_writer": 0,
            "timestamp_gaps": [],
        }
        open_state_path = Path(str(part_path)[:-5] + ".open.json")
        _write_open_state(open_state_path, open_info)
        with self._lock:
            self._open[str(part_path)] = open_info
            self._current_location = str(part_path)
        return str(part_path)

    def note_frame(self, *, source_stamp_ns: int, receive_monotonic_ns: int, dropped: int) -> None:
        with self._lock:
            if not self._current_location:
                return
            info = self._open.get(self._current_location)
            if info is None:
                return
            info["samples_or_frames"] = int(info["samples_or_frames"]) + 1
            info["receive_monotonic_end_ns"] = int(receive_monotonic_ns)
            info["dropped_before_writer"] = int(dropped)
            if source_stamp_ns > 0:
                if not int(info.get("source_stamp_start_ns", 0)):
                    info["source_stamp_start_ns"] = int(source_stamp_ns)
                info["source_stamp_end_ns"] = int(source_stamp_ns)
                info["source_stamp_valid"] = True
            if int(info["samples_or_frames"]) % 30 == 0:
                open_state_path = Path(self._current_location[:-5] + ".open.json")
                _write_open_state(open_state_path, info)

    def finalize_location(self, location: str | Path, *, recovered: bool = False) -> dict[str, Any]:
        part_path = Path(location)
        with self._lock:
            open_info = self._open.pop(str(part_path), None)
            if self._current_location == str(part_path):
                self._current_location = ""
        metadata = self.finalize_part(
            part_path=part_path,
            ledger=self.ledger,
            open_info=open_info,
            recovered=recovered,
        )
        self.finalized_segments += 1
        self.last_finalized = metadata
        return metadata

    def preserve_open_fragments_as_corrupt(self, reason: str) -> int:
        """Preserve fragments from a terminal pipeline failure for diagnosis."""
        with self._lock:
            open_items = list(self._open.items())
            self._open.clear()
            self._current_location = ""

        preserved = 0
        for location, open_info in open_items:
            part_path = Path(location)
            final_path = Path(str(part_path)[:-5])
            open_state_path = Path(str(final_path) + ".open.json")
            if part_path.exists():
                corrupt_path = part_path.with_name(part_path.name + ".corrupt")
                os.replace(part_path, corrupt_path)
                reason_path = corrupt_path.with_name(corrupt_path.name + ".json")
                temp_path = _write_json_temp(reason_path, {
                    "state": "CORRUPT",
                    "source": str(part_path),
                    "reason": str(reason),
                    "open_state": open_info,
                })
                os.replace(temp_path, reason_path)
                preserved += 1
            open_state_path.unlink(missing_ok=True)
            fsync_directory(part_path.parent)
        return preserved

    @staticmethod
    def finalize_part(
        *,
        part_path: Path,
        ledger: SegmentLedger,
        open_info: dict[str, Any] | None = None,
        recovered: bool = False,
    ) -> dict[str, Any]:
        if not part_path.exists() or part_path.stat().st_size <= 0:
            raise ValueError(f"video part is missing or empty: {part_path}")
        final_path = Path(str(part_path)[:-5])
        open_state_path = Path(str(final_path) + ".open.json")
        if open_info is None:
            if not open_state_path.exists():
                raise ValueError(f"video open state is missing: {open_state_path}")
            open_info = json.loads(open_state_path.read_text(encoding="utf-8"))

        with part_path.open("rb") as handle:
            os.fsync(handle.fileno())
        now_ns = time.time_ns()
        start_mono = int(open_info.get("receive_monotonic_start_ns", 0))
        end_mono = int(open_info.get("receive_monotonic_end_ns", 0))
        duration = max(0.0, (end_mono - start_mono) / 1_000_000_000) if end_mono and start_mono else 0.0
        metadata = {
            **open_info,
            "wall_clock_end_ns": now_ns,
            "wall_clock_end_utc": datetime.fromtimestamp(now_ns / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "duration_seconds": duration,
            "bytes": part_path.stat().st_size,
            "sha256": _sha256(part_path),
            "recovered_after_unclean_shutdown": bool(recovered),
        }
        if recovered:
            metadata["source_stamp_valid"] = False
        metadata_path = final_path.with_suffix(".json")
        metadata_temp = _write_json_temp(metadata_path, metadata)
        os.replace(part_path, final_path)
        os.replace(metadata_temp, metadata_path)
        fsync_directory(final_path.parent)
        ledger.upsert_finalized(SegmentRecord.from_metadata(metadata, metadata_path))
        open_state_path.unlink(missing_ok=True)
        fsync_directory(final_path.parent)
        return metadata


def _gst_discoverer_valid(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["gst-discoverer-1.0", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    output = result.stdout.lower()
    return result.returncode == 0 and "video" in output and "error" not in output


def reconcile_video_store(
    data_root: str | Path,
    ledger: SegmentLedger,
    *,
    validator: Callable[[Path], bool] | None = None,
) -> dict[str, int]:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    validate = validator or _gst_discoverer_valid
    stats = {"parts_recovered": 0, "finalized_rebuilt": 0, "corrupt_parts": 0}

    for part_path in sorted(root.rglob("*.mp4.part")):
        try:
            if not validate(part_path):
                raise ValueError("gst-discoverer could not validate interrupted MP4")
            VideoFragmentStore.finalize_part(part_path=part_path, ledger=ledger, recovered=True)
            stats["parts_recovered"] += 1
        except Exception as exc:
            corrupt_path = part_path.with_name(part_path.name + ".corrupt")
            part_path.replace(corrupt_path)
            reason_path = corrupt_path.with_name(corrupt_path.name + ".json")
            temp_path = _write_json_temp(reason_path, {
                "state": "CORRUPT",
                "source": str(part_path),
                "reason": str(exc),
            })
            os.replace(temp_path, reason_path)
            fsync_directory(reason_path.parent)
            stats["corrupt_parts"] += 1

    for metadata_path in sorted(root.rglob("*.json")):
        if metadata_path.name.endswith((".open.json", ".corrupt.json")):
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("kind") != "video" or "segment_id" not in metadata:
                continue
            record = SegmentRecord.from_metadata(metadata, metadata_path)
            if Path(record.local_path).exists() and ledger.get(record.segment_id) is None:
                ledger.upsert_finalized(record)
                stats["finalized_rebuilt"] += 1
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return stats
