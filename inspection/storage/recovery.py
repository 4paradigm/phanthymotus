from __future__ import annotations

import json
from pathlib import Path

from .atomic_writer import AudioSegmentWriter
from .ledger import SegmentLedger
from .models import SegmentRecord


def reconcile_audio_store(data_root: str | Path, ledger: SegmentLedger) -> dict[str, int]:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    stats = {
        "uploading_reset": ledger.reset_uploading_for_recovery(),
        "parts_recovered": 0,
        "finalized_rebuilt": 0,
        "corrupt_parts": 0,
    }

    for part_path in sorted(root.rglob("*.wav.part")):
        try:
            AudioSegmentWriter.recover_part(part_path, ledger)
            stats["parts_recovered"] += 1
        except Exception:
            corrupt_path = part_path.with_name(part_path.name + ".corrupt")
            part_path.replace(corrupt_path)
            reason_path = corrupt_path.with_name(corrupt_path.name + ".json")
            reason_path.write_text(
                json.dumps({"state": "CORRUPT", "source": str(part_path)}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            stats["corrupt_parts"] += 1

    for metadata_path in sorted(root.rglob("*.json")):
        if metadata_path.name.endswith((".open.json", ".corrupt.json")):
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("kind") != "audio" or "segment_id" not in metadata:
                continue
            record = SegmentRecord.from_metadata(metadata, metadata_path)
            if not Path(record.local_path).exists():
                continue
            if ledger.get(record.segment_id) is None:
                ledger.upsert_finalized(record)
                stats["finalized_rebuilt"] += 1
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return stats
