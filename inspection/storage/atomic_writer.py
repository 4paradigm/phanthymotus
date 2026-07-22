from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ledger import SegmentLedger
from .layout import (
    card_storage_slug,
    instance_storage_slug,
    safe_component,
    segment_basename,
    segment_start_ns_from_name,
    utc_hour_partition,
)
from .models import SegmentRecord


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


_safe_component = safe_component


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_temp(final_path: Path, payload: dict[str, Any]) -> Path:
    temp_path = final_path.with_name(f".{final_path.name}.{uuid.uuid4().hex}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return temp_path


def _write_open_state(path: Path, payload: dict[str, Any]) -> None:
    temp_path = _write_json_temp(path, payload)
    os.replace(temp_path, path)
    fsync_directory(path.parent)


class AudioSegmentWriter:
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
        segment_seconds: int,
        max_segment_bytes: int = 4 * 1024 * 1024,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
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
        self.segment_seconds = int(segment_seconds)
        self.max_segment_bytes = max(1, int(max_segment_bytes))
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.sample_width = int(sample_width)
        self._sequence = 0
        self._raw_handle = None
        self._part_path: Path | None = None
        self._open_state_path: Path | None = None
        self._open_info: dict[str, Any] | None = None
        self._samples = 0
        self._source_stamp_start_ns = 0
        self._source_stamp_end_ns = 0
        self._dropped_before_writer = 0
        self.finalized_segments = 0
        self.local_bytes = 0
        self.last_finalized: dict[str, Any] | None = None

    def _begin_segment(self, source_stamp_ns: int) -> None:
        wall_start_ns = time.time_ns()
        utc = datetime.fromtimestamp(wall_start_ns / 1_000_000_000, tz=timezone.utc)
        directory = self.data_root / self.storage_card_slug / self.storage_instance_slug / utc_hour_partition(wall_start_ns)
        directory.mkdir(parents=True, exist_ok=True)
        basename = segment_basename(wall_start_ns, self._sequence, "wav")
        self._sequence += 1
        final_path = directory / basename
        self._part_path = final_path.with_name(final_path.name + ".part")
        self._open_state_path = final_path.with_name(final_path.name + ".open.json")
        self._raw_handle = self._part_path.open("xb", buffering=0)
        self._samples = 0
        self._source_stamp_start_ns = source_stamp_ns if source_stamp_ns > 0 else 0
        self._source_stamp_end_ns = self._source_stamp_start_ns
        self._open_info = {
            "schema_version": "1.0",
            "segment_id": f"seg-{uuid.uuid4().hex}",
            "kind": "audio",
            "device_id": self.device_id,
            "card_id": self.card_id,
            "instance_id": self.instance_id,
            "storage_card_slug": self.storage_card_slug,
            "storage_instance_slug": self.storage_instance_slug,
            "storage_time_partition": utc_hour_partition(wall_start_ns),
            "input_topic": self.input_topic,
            "format": "audio/pcm-16k",
            "ros_type": "audio_msgs/msg/AudioChunk",
            "session_id": self.session_id,
            "sequence": self._sequence - 1,
            "wall_clock_start_ns": wall_start_ns,
            "wall_clock_start_utc": utc.isoformat().replace("+00:00", "Z"),
            "receive_monotonic_start_ns": time.monotonic_ns(),
            "source_stamp_start_ns": self._source_stamp_start_ns,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "max_segment_bytes": self.max_segment_bytes,
            "dropped_before_writer": self._dropped_before_writer,
        }
        _write_open_state(self._open_state_path, self._open_info)

    def write_chunk(
        self,
        pcm: bytes,
        *,
        source_stamp_ns: int = 0,
        dropped_before_writer: int | None = None,
    ) -> dict[str, Any] | None:
        if not pcm:
            return None
        if dropped_before_writer is not None:
            self._dropped_before_writer = max(self._dropped_before_writer, int(dropped_before_writer))
        frame_width = self.channels * self.sample_width
        if len(pcm) % frame_width:
            raise ValueError(f"PCM chunk length {len(pcm)} is not aligned to frame width {frame_width}")
        if self._raw_handle is None:
            self._begin_segment(source_stamp_ns)
        assert self._raw_handle is not None
        self._raw_handle.write(pcm)
        self._samples += len(pcm) // frame_width
        if source_stamp_ns > 0:
            if self._source_stamp_start_ns == 0:
                self._source_stamp_start_ns = source_stamp_ns
            self._source_stamp_end_ns = source_stamp_ns
        current_bytes = self._samples * frame_width
        if (
            self._samples >= self.segment_seconds * self.sample_rate
            or current_bytes >= self.max_segment_bytes
        ):
            return self.finalize()
        return None

    def finalize(self, *, recovered: bool = False) -> dict[str, Any] | None:
        if self._raw_handle is None or self._part_path is None or self._open_info is None:
            return None
        self._raw_handle.flush()
        os.fsync(self._raw_handle.fileno())
        self._raw_handle.close()
        self._raw_handle = None
        self._open_info.update({
            "samples_or_frames": self._samples,
            "source_stamp_start_ns": self._source_stamp_start_ns,
            "source_stamp_end_ns": self._source_stamp_end_ns,
            "source_stamp_valid": self._source_stamp_start_ns > 0 and not recovered,
            "receive_monotonic_end_ns": time.monotonic_ns(),
            "recovered_after_unclean_shutdown": recovered,
            "dropped_before_writer": self._dropped_before_writer,
        })
        metadata = self._finalize_part(
            part_path=self._part_path,
            open_info=self._open_info,
            ledger=self.ledger,
        )
        self.finalized_segments += 1
        self.local_bytes += int(metadata["bytes"])
        self.last_finalized = metadata
        self._part_path = None
        self._open_state_path = None
        self._open_info = None
        self._samples = 0
        return metadata

    def close(self) -> dict[str, Any] | None:
        return self.finalize()

    @staticmethod
    def _finalize_part(*, part_path: Path, open_info: dict[str, Any], ledger: SegmentLedger) -> dict[str, Any]:
        final_path = Path(str(part_path)[:-5])
        metadata_path = final_path.with_suffix(".json")
        frame_width = int(open_info.get("channels", 1)) * int(open_info.get("sample_width", 2))
        raw_size = part_path.stat().st_size
        aligned_size = raw_size - (raw_size % frame_width)
        if aligned_size <= 0:
            raise ValueError(f"audio part is empty or unaligned: {part_path}")
        if aligned_size != raw_size:
            with part_path.open("r+b") as handle:
                handle.truncate(aligned_size)
                handle.flush()
                os.fsync(handle.fileno())

        wav_temp = final_path.with_name(f".{final_path.name}.{uuid.uuid4().hex}.finalizing")
        with wav_temp.open("wb") as output:
            with wave.open(output, "wb") as wav_file:
                wav_file.setnchannels(int(open_info.get("channels", 1)))
                wav_file.setsampwidth(int(open_info.get("sample_width", 2)))
                wav_file.setframerate(int(open_info.get("sample_rate", 16000)))
                with part_path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        wav_file.writeframesraw(chunk)
            output.flush()
            os.fsync(output.fileno())

        wav_size = wav_temp.stat().st_size
        wav_sha256 = _sha256(wav_temp)
        now_ns = time.time_ns()
        sample_rate = int(open_info.get("sample_rate", 16000))
        samples = aligned_size // frame_width
        metadata = {
            **open_info,
            "wall_clock_end_ns": now_ns,
            "wall_clock_end_utc": datetime.fromtimestamp(now_ns / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "duration_seconds": samples / sample_rate,
            "bytes": wav_size,
            "sha256": wav_sha256,
            "samples_or_frames": samples,
            "dropped_before_writer": int(open_info.get("dropped_before_writer", 0)),
            "timestamp_gaps": list(open_info.get("timestamp_gaps", [])),
        }
        metadata_temp = _write_json_temp(metadata_path, metadata)
        os.replace(wav_temp, final_path)
        os.replace(metadata_temp, metadata_path)
        fsync_directory(final_path.parent)

        record = SegmentRecord.from_metadata(metadata, metadata_path)
        ledger.upsert_finalized(record)

        open_state_path = final_path.with_name(final_path.name + ".open.json")
        part_path.unlink(missing_ok=True)
        open_state_path.unlink(missing_ok=True)
        fsync_directory(final_path.parent)
        return metadata

    @classmethod
    def recover_part(cls, part_path: str | Path, ledger: SegmentLedger) -> dict[str, Any]:
        part = Path(part_path)
        final_path = Path(str(part)[:-5])
        open_state_path = final_path.with_name(final_path.name + ".open.json")
        if final_path.exists() and final_path.with_suffix(".json").exists():
            with final_path.with_suffix(".json").open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            ledger.upsert_finalized(SegmentRecord.from_metadata(metadata, final_path.with_suffix(".json")))
            part.unlink(missing_ok=True)
            open_state_path.unlink(missing_ok=True)
            fsync_directory(final_path.parent)
            return metadata
        if open_state_path.exists():
            with open_state_path.open(encoding="utf-8") as handle:
                open_info = json.load(handle)
        else:
            wall_start_ns = segment_start_ns_from_name(final_path)
            partition = final_path.parent.name
            if partition.startswith("utc-hour="):
                storage_instance = final_path.parents[1].name
                storage_card = final_path.parents[2].name
            else:
                storage_instance = final_path.parents[2].name
                storage_card = final_path.parents[3].name
            open_info = {
                "schema_version": "1.0",
                "segment_id": f"seg-recovered-{uuid.uuid4().hex}",
                "kind": "audio",
                "device_id": "unknown",
                "card_id": "audioinspector",
                "instance_id": "unknown",
                "storage_card_slug": storage_card,
                "storage_instance_slug": storage_instance,
                "storage_time_partition": partition,
                "input_topic": "",
                "format": "audio/pcm-16k",
                "ros_type": "audio_msgs/msg/AudioChunk",
                "session_id": "recovered",
                "sequence": 0,
                "wall_clock_start_ns": wall_start_ns,
                "wall_clock_start_utc": datetime.fromtimestamp(wall_start_ns / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                "receive_monotonic_start_ns": 0,
                "source_stamp_start_ns": 0,
                "sample_rate": 16000,
                "channels": 1,
                "sample_width": 2,
            }
        open_info.update({
            "source_stamp_valid": False,
            "source_stamp_end_ns": 0,
            "receive_monotonic_end_ns": 0,
            "recovered_after_unclean_shutdown": True,
        })
        return cls._finalize_part(part_path=part, open_info=open_info, ledger=ledger)
