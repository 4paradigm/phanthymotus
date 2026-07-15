from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from urllib.request import urlretrieve


MODEL_FILES = ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt")
MAX_BUNDLE_BYTES = 15 * 1024 * 1024


def download_model(
    base_url: str,
    output_dir: str,
    filenames=MODEL_FILES,
    max_bundle_bytes=MAX_BUNDLE_BYTES,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="ocr-model-", dir=output.parent
    ) as staging_dir:
        staging = Path(staging_dir)
        for filename in filenames:
            staged_file = staging / filename
            urlretrieve(f"{base_url.rstrip('/')}/{filename}", staged_file)
            if staged_file.stat().st_size == 0:
                raise ValueError(
                    f"Downloaded OCR model file is empty: {filename}"
                )

        total = sum((staging / name).stat().st_size for name in filenames)
        if total > max_bundle_bytes:
            raise ValueError(
                f"OCR model bundle is {total} bytes, exceeds 15 MiB limit"
            )

        for filename in filenames:
            os.replace(staging / filename, output / filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    download_model(args.base_url, args.output_dir)


if __name__ == "__main__":
    main()
