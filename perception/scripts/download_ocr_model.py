#!/usr/bin/env python3
"""下载 PP-OCRv6 Tiny 离线模型包。"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen


MODEL_FILES = ("det.onnx", "rec.onnx", "inference.yml")
MAX_BUNDLE_BYTES = 15 * 1024 * 1024
DOWNLOAD_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_DELAY = 3


def download_file(url: str, destination: Path) -> None:
    """带超时和重试下载单个文件。"""
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
                data = response.read()
            destination.write_bytes(data)
            if destination.stat().st_size == 0:
                raise ValueError(f"Downloaded file is empty: {url}")
            return
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if destination.exists():
                destination.unlink()
            if attempt < MAX_RETRIES:
                print(
                    f"  Retry {attempt}/{MAX_RETRIES} after error: {exc}",
                    flush=True,
                )
                time.sleep(RETRY_DELAY)
    assert last_error is not None
    raise last_error


def download_model(
    base_url: str,
    output_dir: str,
    filenames: Tuple[str, ...] = MODEL_FILES,
    max_bundle_bytes: int = MAX_BUNDLE_BYTES,
) -> None:
    """完整下载模型包，通过校验后原子替换目标文件。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="ocr-model-",
        dir=output.parent,
    ) as staging_dir:
        staging = Path(staging_dir)
        for filename in filenames:
            staged_file = staging / filename
            url = f"{base_url.rstrip('/')}/{filename}"
            print(f"Downloading {url}", flush=True)
            download_file(url, staged_file)
            print(f"  OK ({staged_file.stat().st_size} bytes)", flush=True)

        total = sum((staging / name).stat().st_size for name in filenames)
        if total > max_bundle_bytes:
            raise ValueError(
                f"OCR model bundle is {total} bytes, exceeds 15 MiB limit"
            )

        for filename in filenames:
            os.replace(staging / filename, output / filename)

    print(f"OCR model download complete! Total: {total} bytes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    download_model(args.base_url, args.output_dir)


if __name__ == "__main__":
    main()
