"""Reject mismatched robot package/runtime before opening the Motion API."""
import hashlib
import json
import os
from pathlib import Path
import sys

import mujoco

if mujoco.__version__ != "3.16.0":
    raise RuntimeError("G1 package requires MuJoCo 3.16.0")
manifest = json.loads(Path("/profiles/manifest.json").read_text())
for name, expected in manifest["sha256"].items():
    if hashlib.sha256((Path("/profiles") / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError("G1 package checksum mismatch")
os.execv(sys.executable, [sys.executable, "-m", "motion_service.server", "--host", "0.0.0.0",
                         "--port", "15732", "--data-dir", "/data", "--profiles", "/profiles/profiles.json"])
