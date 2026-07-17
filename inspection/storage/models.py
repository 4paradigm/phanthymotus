from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class SegmentState(str, Enum):
    FINALIZED = "FINALIZED"
    UPLOADING = "UPLOADING"
    UPLOADED_VERIFIED = "UPLOADED_VERIFIED"
    RETENTION_ELIGIBLE = "RETENTION_ELIGIBLE"
    PURGED_LOCAL = "PURGED_LOCAL"
    CORRUPT = "CORRUPT"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class SegmentRecord:
    segment_id: str
    kind: str
    card_id: str
    instance_id: str
    local_path: str
    metadata_path: str
    size: int
    sha256: str
    state: SegmentState = SegmentState.FINALIZED
    object_key: str = ""
    created_at_ns: int = 0
    last_error: str = ""

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any], metadata_path: Path) -> "SegmentRecord":
        extension = ".wav" if metadata.get("kind") == "audio" else ".mp4"
        local_path = metadata_path.with_suffix(extension)
        return cls(
            segment_id=str(metadata["segment_id"]),
            kind=str(metadata["kind"]),
            card_id=str(metadata["card_id"]),
            instance_id=str(metadata["instance_id"]),
            local_path=str(local_path),
            metadata_path=str(metadata_path),
            size=int(metadata["bytes"]),
            sha256=str(metadata["sha256"]),
            state=SegmentState.FINALIZED,
            created_at_ns=int(metadata.get("wall_clock_start_ns", 0)),
        )
