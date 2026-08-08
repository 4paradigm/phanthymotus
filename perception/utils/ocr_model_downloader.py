from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


MODEL_FILES = ("det.mnn", "rec.mnn", "keys.txt")
MAX_BUNDLE_BYTES = 48 * 1024 * 1024
DOWNLOAD_TIMEOUT = 120  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 3  # seconds


def download_file(url: str, dest: Path) -> None:
    """Download a file with timeout and retry logic.

    Uses urlopen with timeout then writes to file (Python 3.8 compat).
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
                data = response.read()
            dest.write_bytes(data)
            size = dest.stat().st_size
            if size == 0:
                raise ValueError(f"Downloaded file is empty: {url}")
            return
        except (URLError, TimeoutError, OSError, ValueError) as e:
            last_error = e
            if dest.exists():
                dest.unlink()
            if attempt < MAX_RETRIES:
                print(f"  Retry {attempt}/{MAX_RETRIES} after error: {e}", flush=True)
                time.sleep(RETRY_DELAY)
    assert last_error is not None
    raise last_error


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(
    base_url: str,
    output_dir: str,
    filenames=MODEL_FILES,
    max_bundle_bytes=MAX_BUNDLE_BYTES,
    checksums: dict[str, str] | None = None,
) -> None:
    checksums = checksums or {}
    unknown_checksums = set(checksums) - set(filenames)
    if unknown_checksums:
        raise ValueError(
            f"checksums provided for unknown files: {sorted(unknown_checksums)}"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="ocr-model-", dir=output.parent
    ) as staging_dir:
        staging = Path(staging_dir)
        for filename in filenames:
            staged_file = staging / filename
            url = f"{base_url.rstrip('/')}/{filename}"
            print(f"Downloading {url}", flush=True)
            download_file(url, staged_file)
            expected = checksums.get(filename)
            if expected:
                actual = file_sha256(staged_file)
                if actual.lower() != expected.lower():
                    raise ValueError(
                        f"SHA256 mismatch for {filename}: expected "
                        f"{expected}, got {actual}"
                    )
            print(f"  OK ({staged_file.stat().st_size} bytes)", flush=True)

        total = sum((staging / name).stat().st_size for name in filenames)
        if total > max_bundle_bytes:
            raise ValueError(
                f"OCR model bundle is {total} bytes, exceeds configured "
                f"{max_bundle_bytes} byte limit"
            )

        for filename in filenames:
            os.replace(staging / filename, output / filename)

    print("OCR model download complete!", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--filenames", nargs="+", default=list(MODEL_FILES))
    parser.add_argument(
        "--sha256",
        action="append",
        default=[],
        metavar="FILE=DIGEST",
        help="expected SHA256; may be repeated",
    )
    args = parser.parse_args()
    checksums = {}
    for value in args.sha256:
        filename, separator, digest = value.partition("=")
        if not separator or len(digest) != 64:
            raise ValueError(f"invalid --sha256 value: {value}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"invalid --sha256 digest: {digest}") from exc
        checksums[filename] = digest.lower()
    download_model(
        args.base_url,
        args.output_dir,
        filenames=tuple(args.filenames),
        checksums=checksums,
    )


if __name__ == "__main__":
    main()
