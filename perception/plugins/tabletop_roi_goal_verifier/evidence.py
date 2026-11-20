"""Save one session export into a new evidence directory; no session token on disk."""

import base64
import hashlib
import json
import tempfile
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
    images = exported.get("roi_evidence", {})
    complete = all(
        images.get(k) for k in ("before_target_jpeg_base64", "after_target_jpeg_base64")
    )
    need(
        result.get("result") not in ("PASS", "FAIL") or bool(complete),
        "EVIDENCE_INCOMPLETE",
    )
    exported["test_record"]["evidence_complete"] = bool(complete)
    # Build privately, then publish one complete directory atomically. Any
    # decode/write/checksum error removes the temporary partial run.
    with tempfile.TemporaryDirectory(prefix=".incomplete-", dir=root) as temporary:
        staging = Path(temporary)
        (staging / "session.json").write_text(
            json.dumps(exported, ensure_ascii=False, indent=2)
        )
        (staging / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        for key, filename in [
            ("before_target_jpeg_base64", "before-roi.jpg"),
            ("after_target_jpeg_base64", "after-roi.jpg"),
        ]:
            if images.get(key):
                data = base64.b64decode(images[key], validate=True)
                import cv2
                import numpy as np

                need(
                    cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    is not None,
                    "EVIDENCE_IMAGE_INVALID",
                )
                (staging / filename).write_bytes(data)
        sums = [
            hashlib.sha256(file.read_bytes()).hexdigest() + "  " + file.name
            for file in sorted(staging.iterdir())
        ]
        (staging / "SHA256SUMS").write_text("\n".join(sums) + "\n")
        staging.rename(directory)
    return {
        "state": "saved",
        "directory": str(directory),
        "expected": label,
        "actual": result["result"],
        "provenance": provenance,
    }
