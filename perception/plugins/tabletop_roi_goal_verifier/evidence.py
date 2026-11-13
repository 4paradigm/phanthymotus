"""Save one session export into a new evidence directory; no session token on disk."""

import base64
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .contracts import need


def save_session(exported, root, label, notes=""):
    need(label in ("PASS", "FAIL", "INCONCLUSIVE"), "INVALID_TEST_LABEL")
    need(isinstance(notes, str) and len(notes) <= 2000, "INVALID_NOTES")
    exported = deepcopy(exported)
    exported.pop("session_token", None)
    need(isinstance(exported.get("result"), dict), "VERIFICATION_REQUIRED")
    result = exported["result"]
    provenance = (exported.get("before") or {}).get("provenance")
    exported["test_record"] = {
        "expected": label,
        "actual": result["result"],
        "matches_expected": label == result["result"],
        "operator_notes": notes,
        "provenance": provenance,
        "hardware_acceptance": False,
        "note": "Real provenance alone does not prove hardware identity or complete field acceptance.",
    }
    root = Path(root).resolve(strict=True)
    name = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + label
        + "-"
        + uuid4().hex[:8]
    )
    directory = root / name
    directory.mkdir(mode=0o700)
    (directory / "session.json").write_text(
        json.dumps(exported, ensure_ascii=False, indent=2)
    )
    (directory / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    images = exported.get("roi_evidence", {})
    for key, filename in [
        ("before_target_jpeg_base64", "before-roi.jpg"),
        ("after_target_jpeg_base64", "after-roi.jpg"),
    ]:
        if images.get(key):
            (directory / filename).write_bytes(
                base64.b64decode(images[key], validate=True)
            )
    sums = []
    for file in sorted(directory.iterdir()):
        sums.append(hashlib.sha256(file.read_bytes()).hexdigest() + "  " + file.name)
    (directory / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return {
        "state": "saved",
        "directory": str(directory),
        "expected": label,
        "actual": result["result"],
        "provenance": provenance,
    }
