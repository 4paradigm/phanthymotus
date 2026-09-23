#!/usr/bin/env python3
"""Materialize pinned G1 collision assets before building/starting ActuCore.

Never invoked by a control loop. The runtime independently verifies the same
hash manifest and refuses to initialize if an asset is absent or corrupt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen


SOURCE = (
    "https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/"
    "public/teleop/g1-collision/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b/"
)
MAX_ASSET_BYTES = 16 * 1024 * 1024


def verified(path, expected):
    if not path.is_file() or path.stat().st_size > MAX_ASSET_BYTES:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest() == expected


def fetch(manifest_dir, destination, check_only=False):
    hashes = json.loads((manifest_dir / "sha256.json").read_text())
    # Only the checked-in flat mesh manifest is accepted, never arbitrary paths.
    for name, digest in hashes.items():
        if (Path(name).name != name or not name.endswith(".STL") or
                len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid collision asset manifest")
    if not check_only:
        destination.mkdir(parents=True, exist_ok=True)
    for name, digest in hashes.items():
        target = destination / name
        if verified(target, digest):
            continue
        if check_only:
            raise ValueError("missing or corrupt collision mesh: " + name)
        staging = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination, suffix=".part", delete=False) as out:
                staging = Path(out.name)
                with urlopen(SOURCE + name, timeout=30) as source:
                    total = 0
                    while block := source.read(65536):
                        total += len(block)
                        if total > MAX_ASSET_BYTES:
                            raise ValueError("collision asset exceeds size limit: " + name)
                        out.write(block)
            if not verified(staging, digest):
                raise ValueError("collision asset SHA256 mismatch: " + name)
            staging.replace(target)
        finally:
            if staging is not None:
                staging.unlink(missing_ok=True)
    return len(hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path(__file__).resolve().parents[1] /
                        "actucore/plugins/teleop/models/g1_collision")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--check", action="store_true", help="verify existing files without network or writes")
    args = parser.parse_args()
    count = fetch(args.manifest_dir, args.destination or args.manifest_dir, args.check)
    print(f"G1 COLLISION PASS: {count} assets verified")


if __name__ == "__main__":
    main()
