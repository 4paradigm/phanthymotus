#!/usr/bin/env python3
"""Download versioned VITS2 resources used by the Jetson image."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_BASE_URL = (
    "http://172.28.4.81:34567/liaoqianqian/models/vits2-mix"
)

RESOURCES = {
    "deps": (
        "jetson-prebuilt-deps.tar.gz",
        "2c36652e25e553d5e8c650fb6b541c575551f47d447e342a6440625370a6a23d",
    ),
    "assets": (
        "vits2-mix-assets.tar.gz",
        "d299ca41ebdd73d884399738db21664b7bc79ccdbcb628b56d29607aaf3556e0",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_resource(resource: str, base_url: str, output_dir: Path) -> Path:
    filename, expected_sha256 = RESOURCES[resource]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{filename}.", dir=str(output_dir), delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        url = f"{base_url.rstrip('/')}/{filename}"
        print(f"Downloading {url}", flush=True)
        urlretrieve(url, str(temporary_path))
        actual_sha256 = sha256(temporary_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA256 mismatch for {filename}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        temporary_path.replace(destination)
        temporary_path = None
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resource", choices=sorted(RESOURCES))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/vits2"))
    args = parser.parse_args()
    print(download_resource(args.resource, args.base_url, args.output_dir))


if __name__ == "__main__":
    main()
