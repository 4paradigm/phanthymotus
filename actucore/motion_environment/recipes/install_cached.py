"""Install only administrator-prepared original wheels before dependency resolution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path("/opt/wheels")
manifest = json.loads((root / "manifest.json").read_text())
files = []
for entry in manifest["wheels"]:
    if Path(entry["filename"]).name != entry["filename"]:
        raise RuntimeError("invalid cached wheel filename")
    path = root / entry["filename"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != entry["sha256"]:
        raise RuntimeError("cached wheel does not match its official SHA256")
    files.append(str(path))
if files:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", *files], check=True)
