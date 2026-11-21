"""Explicit opt-in, verified downloads for the standalone face-effects Card.

The Card's runtime never fetches models. Only this CLI downloads weights.
The shared downloader provides pinned size/SHA256 validation, file locking,
atomic staging, retries, and byte-progress callbacks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

# Direct invocation from the repository root does not otherwise expose
# perception/utils on sys.path.
PERCEPTION_ROOT = Path(__file__).resolve().parents[2]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from utils.model_downloader import ensure_verified_bundle
from utils.model_progress import fetch_status


def _verify(path: Path, spec: dict, name: str) -> None:
    """Reject existing, truncated or modified model bytes without replacing them."""
    expected_size = int(spec["bytes"])
    expected_sha = spec["sha256"]
    try:
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(f"{name}: model size mismatch: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha:
            raise RuntimeError(f"{name}: model SHA256 mismatch: {path}")
    except OSError as error:
        raise RuntimeError(f"{name}: cannot verify model: {path}") from error


def fetch_one(name: str, spec: dict, root: Path, on_status=None) -> Path:
    """Provision the verified upstream file under the Card's local model alias.

    The checked upstream file is cached under .verified/<model-name>/.
    Its upstream basename differs from the Card's documented local name
    (e.g. face_detection_yunet_2023mar.onnx vs yunet.onnx).
    Linking on the same filesystem keeps one copy without changing consumers.
    """
    local_filename = spec["file"]
    if not local_filename or Path(local_filename).name != local_filename:
        raise ValueError(f"{name}: invalid local model filename")
    url = spec["url"]
    parsed = urlsplit(url)
    remote_filename = Path(parsed.path).name
    if (parsed.scheme != "https" or not parsed.netloc or not remote_filename
            or parsed.query or parsed.fragment):
        raise ValueError(f"{name}: expected an HTTPS model URL ending in a filename")

    target = root / local_filename
    if target.exists() or target.is_symlink():
        _verify(target, spec, name)
        if on_status:
            on_status(f"{name}: existing model verified")
        return target

    # The shared verified downloader owns cache locking, retries and progress.
    cache_dir = root / ".verified" / name
    progress_cb, _ = fetch_status(on_status, name)
    cache_paths = ensure_verified_bundle(
        name=f"face_effects_live_{name}",
        model_dir=str(cache_dir),
        base_url=url.rsplit("/", 1)[0],
        files={remote_filename: {
            "size": int(spec["bytes"]),
            "sha256": spec["sha256"],
        }},
        progress_cb=progress_cb,
    )
    verified_source = Path(cache_paths[remote_filename])
    _verify(verified_source, spec, name)

    # Atomic no-overwrite publication of the documented alias. A second
    # concurrent invocation may win; it must publish the same pinned bytes.
    try:
        os.link(verified_source, target)
    except FileExistsError:
        pass
    _verify(target, spec, name)
    if on_status:
        on_status(f"{name}: downloaded and verified ({spec['license']})")
    return target


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--models", nargs="+", choices=["yunet", "osf0", "osf3", "mediapipe"],
        default=["yunet", "osf0"],
    )
    args = parser.parse_args(argv)
    manifest = PERCEPTION_ROOT / "plugins" / "face_effects_live" / "models.json"
    models = json.loads(manifest.read_text(encoding="utf-8"))["models"]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    for name in args.models:
        def status(message):
            print(message, flush=True)

        fetch_one(name, models[name], root, on_status=status)


if __name__ == "__main__":
    main()
