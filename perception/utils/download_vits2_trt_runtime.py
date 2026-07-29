#!/usr/bin/env python3
"""Download and verify the versioned JetPack 6 VITS2 TensorRT runtime bundle."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlopen


DEFAULT_URL = (
    "http://172.28.4.81:34567/liaoqianqian/models/vits2-mix/"
    "vits2-mix-jp6-runtime.tar.gz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: Path, expected_sha256: str) -> None:
    if len(expected_sha256) != 64:
        raise ValueError("A 64-character VITS2 runtime SHA256 is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", dir=output.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urlopen(url) as response:
                while block := response.read(1024 * 1024):
                    temporary.write(block)
        actual = sha256(temporary_path)
        if actual != expected_sha256.lower():
            raise RuntimeError(
                f"SHA256 mismatch for {output.name}: expected "
                f"{expected_sha256.lower()}, got {actual}"
            )
        temporary_path.replace(output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    download(args.url, args.output, args.sha256)


if __name__ == "__main__":
    main()
